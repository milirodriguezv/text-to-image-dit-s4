import os
import hashlib

import json
import matplotlib.pyplot as plt

import csv
from pathlib import Path
from PIL import Image
from torchvision import transforms
from collections import defaultdict

import torch
from diffusers import AutoencoderKL

import numpy as np

from text_utils import encode_texts, load_clip

def load_images(
    annotation_file: str,
    captions_file: str,
    images_dir: str,
    max_images: int | None = None
) -> list:
    """
    Load image metadata, person bounding boxes and captions.

    Args:
        annotation_file: Path to the COCO person keypoints JSON file.
        captions_file: Path to the COCO captions JSON file.
        images_dir: Directory containing the COCO images.
        max_images: Maximum number of images to load.
            If None, load all eligible images.

    Returns:
        A list of dictionaries containing:
            - image_id: COCO image identifier.
            - image_path: Path to the image file.
            - width: Image width in pixels.
            - height: Image height in pixels.
            - bboxes: List of person bounding boxes in COCO format.
            - captions: List of five captions associated with the image.
    """

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(captions_file, "r", encoding="utf-8") as f:
        captions_data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    # Group bounding boxes by image_id.
    bboxes_by_image = {}

    for annotation in annotations:
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]

        bboxes_by_image.setdefault(image_id, []).append(bbox)

    # Group captions by image_id.
    captions_by_image = {}

    for annotation in captions_data["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"]

        captions_by_image.setdefault(image_id, []).append(caption)

    image_data = []

    for image in images:
        image_id = image["id"]
        bboxes = bboxes_by_image.get(image_id, [])
        captions = captions_by_image.get(image_id, [])

        if not bboxes or len(captions) < 5:
            continue

        image_path = Path(images_dir) / image["file_name"]

        if not image_path.is_file():
            continue

        image_data.append({
            "image_id": image_id,
            "image_path": image_path,
            "width": image["width"],
            "height": image["height"],
            "bboxes": bboxes,
            "captions": captions[:5]
        })

        if max_images is not None and len(image_data) >= max_images:
            break

    return image_data

def load_bbox(bboxes: list[list[int | float]]) -> list[float]:
    """
    Computes the union of multiple bounding boxes.

    Given a list of bounding boxes in COCO format, calculates the smallest
    bounding box that contains all of them.

    Args:
        bboxes (list): A list of bounding boxes, each represented as
            [x_min, y_min, width, height], where (x_min, y_min) denotes
            the top-left corner and width and height define the box dimensions.

    Returns:
        list: The union bounding box in COCO format
            [x_min, y_min, width, height], enclosing all input bounding boxes.
    """

    x_min = float('inf')
    y_min = float('inf')
    x_max = float('-inf')
    y_max = float('-inf')

    for x, y, width, height in bboxes:
        # Calcular las esquinas opuestas
        x_w = x + width
        y_h = y + height
        

        # Comparar para la esquina superior
        x_min = min(x_min, x)
        y_min = min(y_min, y)

        # Comparar para la esquina inferior
        x_max = max(x_max, x_w)
        y_max = max(y_max, y_h)


    union_width = x_max - x_min
    union_height = y_max - y_min

    bbox_final = [x_min, y_min, union_width, union_height]
            
    return bbox_final


def expand_bbox(bbox: list, margin: float = 0.2):
    """
    Expand a bounding box by adding a relative margin around it.

    The expansion is applied symmetrically around the center of
    the bounding box.

    Args:
        bbox: Bounding box in COCO format [x_min, y_min, width, height].
        margin: Relative increase in width and height.
            For example, 0.2 increases both dimensions by 20%.

    Returns:
        An expanded bounding box in COCO format
        [x_min, y_min, width, height].
    """
    x_min, y_min, width, height = bbox

    delta_x = width * margin / 2
    delta_y = height * margin / 2

    expanded_bbox = [
        x_min - delta_x,
        y_min - delta_y,
        width * (1 + margin),
        height * (1 + margin)
    ]

    return expanded_bbox

def make_square(bbox: list):
    """
    Convert a bounding box into a square by expanding its smaller dimension.

    Args:
        bbox: Bounding box in COCO format
            [x_min, y_min, width, height].

    Returns:
        A square bounding box in COCO format
        [x_min, y_min, side, side].
    """
    x_min, y_min, width, height = bbox

    side = max(width, height)

    delta_x = (side - width) / 2
    delta_y = (side - height) / 2

    square_bbox = [
        x_min - delta_x,
        y_min - delta_y,
        side,
        side
    ]

    return square_bbox

def clip_bbox(bbox: list, image_width: int, image_height: int):
    """
    Adjust a bounding box to the image boundaries.

    Args:
        bbox: Bounding box in COCO format
            [x_min, y_min, width, height].
        image_width: Width of the original image in pixels.
        image_height: Height of the original image in pixels.

    Returns:
        A bounding box [x_min, y_min, width, height] contained
        within the image boundaries.
    """
    x_min, y_min, width, height = bbox

    x_max = x_min + width
    y_max = y_min + height

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_width, x_max)
    y_max = min(image_height, y_max)

    clipped_bbox = [
        x_min,
        y_min,
        x_max - x_min,
        y_max - y_min
    ]

    return clipped_bbox


def preprocess_image(
    image: Image.Image,
    bboxes: list,
    margin: float = 0.2,
    output_size: int = 256
):
    """
    Crop an image around the union of its bounding boxes,
    make the crop square, resize it and normalize its pixels.

    Args:
        image: Original PIL image.
        bboxes: List of bounding boxes in COCO format.
        margin: Relative margin added around the union.
        output_size: Width and height of the output image.

    Returns:
        A normalized image tensor with shape (3, output_size, output_size)
        and values in [-1, 1].
    """
    if not bboxes:
        raise ValueError("The image has no bounding boxes.")

    image = image.convert("RGB")
    image_width, image_height = image.size

    # 1. Calculate the union of the bounding boxes.
    bbox = load_bbox(bboxes)

    bbox = expand_bbox(bbox, margin)

    bbox = make_square(bbox)

    bbox = clip_bbox(bbox, image_width, image_height)

    # Convert coordinates to integers.
    x_min, y_min, width, height = bbox

    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_min + width)
    y_max = int(y_min + height)

    # Cortar la imagen
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_image = cropped_image.resize(
        (output_size, output_size),
        Image.Resampling.BICUBIC
    )

    # Convertir a tensor y normalizarla [-1,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    return transform(cropped_image)


def test_preprocess_image(image_path: str, bboxes: list):
    """
    Display the original image and its preprocessed crop.
    """
    image = Image.open(image_path).convert("RGB")

    tensor = preprocess_image(image, bboxes)

    # Convert the normalized tensor back to an image for visualization.
    display_tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    display_image = display_tensor.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(display_image)
    axes[1].set_title("Preprocessed image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("preprocessing_test.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved visualization to preprocessing_test.png")

def loadVAE(device: str = "cpu"):
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-ema"
    ).to(device)

    vae.eval()

    return vae

def encode_image(vae, image, device="cpu"):
    # Añadir dimensión del batch: (3, 256, 256) -> (1, 3, 256, 256)
    image = image.unsqueeze(0).to(device)

    # Codificar la imagen sin calcular gradientes
    with torch.no_grad():
        latent_dist = vae.encode(image).latent_dist

    return latent_dist

def flip_bboxes_horizontal(bboxes, image_width):
    """
    Transforma las bounding boxes para una imagen girada horizontalmente.

    Cada bbox tiene el formato [x, y, width, height].
    """
    flipped_bboxes = []

    for x, y, width, height in bboxes:
        flipped_x = image_width - x - width

        flipped_bboxes.append([
            flipped_x,
            y,
            width,
            height
        ])

    return flipped_bboxes

def extract_latent_stats(vae, image_tensor, device="cpu"):
    """
    Codifica una imagen preprocesada con el VAE.

    Args:
        vae: VAE previamente cargado.
        image_tensor: Tensor con forma (3, 256, 256).
        device: Dispositivo donde se ejecuta el modelo.

    Returns:
        Tensor con forma (8, 32, 32):
        4 canales de media + 4 canales de desviación típica.
    """
    # Añadir dimensión de batch y mover al dispositivo.
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Obtener la distribución latente.
    with torch.no_grad():
        latent_dist = vae.encode(image_tensor).latent_dist

        mean = latent_dist.mean
        std = latent_dist.std

        # Concatenar los canales: (1, 8, 32, 32).
        latent_stats = torch.cat([mean, std], dim=1)

    # Eliminar la dimensión de batch: (8, 32, 32).
    return latent_stats.squeeze(0)


def extract_image_latents(vae, image_data, device="cpu"):
    """
    Extrae los latents de la imagen original y de su flip horizontal.

    Returns:
        Array NumPy con forma (2, 8, 32, 32).
    """
    # Abrir la imagen original.
    image = Image.open(
        image_data["image_path"]
    ).convert("RGB")

    # Obtener las dimensiones originales.
    image_width, image_height = image.size

    # Vista original.
    original_tensor = preprocess_image(
        image=image,
        bboxes=image_data["bboxes"]
    )

    original_latents = extract_latent_stats(
        vae=vae,
        image_tensor=original_tensor,
        device=device
    )

    # Voltear la imagen horizontalmente.
    flipped_image = image.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    # Transformar las bounding boxes para la imagen volteada.
    flipped_bboxes = flip_bboxes_horizontal(
        bboxes=image_data["bboxes"],
        image_width=image_width
    )

    # Preprocesar la vista volteada con las cajas transformadas.
    flipped_tensor = preprocess_image(
        image=flipped_image,
        bboxes=flipped_bboxes
    )

    flipped_latents = extract_latent_stats(
        vae=vae,
        image_tensor=flipped_tensor,
        device=device
    )

    # Apilar ambas vistas.
    latents = torch.stack(
        [original_latents, flipped_latents],
        dim=0
    )

    return latents.cpu().numpy()

def get_image_ids(images):
    return np.asarray(
        [image["image_id"] for image in images],
        dtype=np.int64
    )


def get_dataset_signature(image_ids):
    return hashlib.sha256(
        image_ids.tobytes()
    ).hexdigest()


def load_or_create_progress(progress_path, n_images, signature):
    """
    Carga el progreso anterior o crea uno nuevo.
    El progreso indica qué imágenes están completamente procesadas.
    """
    progress_path = Path(progress_path)
    metadata_path = progress_path.with_suffix(".json")

    if progress_path.exists() and metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if (
            metadata.get("signature") != signature
            or metadata.get("n_images") != n_images
        ):
            raise ValueError(
                f"El progreso de {progress_path.name} "
                "no corresponde al conjunto actual de imágenes. "
                "No se continuará para evitar mezclar datos."
            )

        progress = np.load(progress_path)

        if progress.shape != (n_images,):
            raise ValueError("El archivo de progreso tiene una forma incorrecta.")

        return progress.astype(bool)

    if progress_path.exists() or metadata_path.exists():
        raise ValueError(
            f"El progreso de {progress_path.name} está incompleto. "
            "Revísalo antes de continuar."
        )

    progress = np.zeros(n_images, dtype=bool)

    save_progress(
        progress,
        progress_path,
        metadata_path,
        n_images,
        signature
    )

    return progress


def save_progress(progress, progress_path, metadata_path,
                  n_images, signature):
    """
    Guarda el progreso mediante archivos temporales y reemplazo atómico.
    """
    progress_path = Path(progress_path)
    metadata_path = Path(metadata_path)

    temp_progress = progress_path.with_name(
        progress_path.name + ".tmp.npy"
    )
    temp_metadata = metadata_path.with_name(
        metadata_path.name + ".tmp"
    )

    np.save(temp_progress, progress)

    with open(temp_metadata, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_images": n_images,
                "signature": signature
            },
            f
        )
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_progress, progress_path)
    os.replace(temp_metadata, metadata_path)


def open_or_create_memmap(path, shape, dtype):
    """
    Abre un archivo existente para continuar o crea uno nuevo.
    No sobrescribe silenciosamente archivos con dimensiones distintas.
    """
    path = Path(path)

    if path.exists():
        array = np.load(path, mmap_mode="r+")

        if array.shape != shape or array.dtype != np.dtype(dtype):
            del array
            raise ValueError(
                f"{path} tiene dimensiones o tipo incorrectos. "
                "No se sobrescribirá automáticamente."
            )

        return array

    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=shape
    )


def extract_dataset_latents(images, vae, output_dir, device="cpu"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(images)
    if n == 0:
        raise ValueError("No hay imágenes para procesar.")

    ids = get_image_ids(images)
    signature = get_dataset_signature(ids)

    ids_path = output_dir / "image_ids.npy"

    if ids_path.exists():
        stored_ids = np.load(ids_path)

        if not np.array_equal(stored_ids, ids):
            raise ValueError(
                "Los IDs existentes no coinciden con las imágenes actuales."
            )
    else:
        np.save(ids_path, ids)

    latents = open_or_create_memmap(
        output_dir / "latents.npy",
        (n, 2, 8, 32, 32),
        np.float16
    )

    progress_path = output_dir / "latents_progress.npy"
    metadata_path = progress_path.with_suffix(".json")

    done = load_or_create_progress(
        progress_path, n, signature
    )

    try:
        for i, image_data in enumerate(images):
            if done[i]:
                continue

            print(
                f"Latentes: {i + 1}/{n} "
                f"(ID: {image_data['image_id']})"
            )

            result = extract_image_latents(
                vae=vae,
                image_data=image_data,
                device=device
            )

            if result.shape != (2, 8, 32, 32):
                raise ValueError(
                    f"Shape incorrecto en la imagen {image_data['image_id']}"
                )

            if not np.isfinite(result).all():
                raise ValueError(
                    f"Latentes no finitos en la imagen {image_data['image_id']}"
                )

            latents[i] = result.astype(np.float16)

            # Sincronizar los datos antes de marcar la imagen como terminada.
            latents.flush()
            done[i] = True

            if (i + 1) % 25 == 0 or i == n - 1:
                save_progress(
                    done, progress_path, metadata_path, n, signature
                )
                print(f"Progreso: {done.sum()}/{n}")

        # Guardar también el progreso final.
        latents.flush()
        save_progress(
            done, progress_path, metadata_path, n, signature
        )

    finally:
        del latents

    print(f"Latentes completados: {done.sum()}/{n}")


def extract_dataset_text_embeddings(
    images, tokenizer, text_model, output_dir, device="cpu"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(images)
    if n == 0:
        raise ValueError("No hay imágenes para procesar.")

    ids = get_image_ids(images)
    signature = get_dataset_signature(ids)

    ids_path = output_dir / "image_ids.npy"

    if not ids_path.exists():
        raise FileNotFoundError(
            "No existe image_ids.npy. Ejecuta primero la extracción de latentes."
        )

    stored_ids = np.load(ids_path)

    if not np.array_equal(stored_ids, ids):
        raise ValueError(
            "Los IDs de las imágenes no coinciden con los latentes."
        )

    embeddings = open_or_create_memmap(
        output_dir / "e_pooled.npy",
        (n, 5, 768),
        np.float16
    )

    progress_path = output_dir / "embeddings_progress.npy"
    metadata_path = progress_path.with_suffix(".json")

    done = load_or_create_progress(
        progress_path, n, signature
    )

    try:
        for i, image_data in enumerate(images):
            if done[i]:
                continue

            captions = image_data["captions"]

            if len(captions) != 5:
                raise ValueError(
                    f"La imagen {image_data['image_id']} "
                    "no tiene exactamente cinco captions."
                )

            print(
                f"Embeddings: {i + 1}/{n} "
                f"(ID: {image_data['image_id']})"
            )

            result = encode_texts(
                texts=captions,
                tokenizer=tokenizer,
                text_model=text_model,
                device=device
            )

            if tuple(result.shape) != (5, 768):
                raise ValueError(
                    f"Shape incorrecto en la imagen {image_data['image_id']}"
                )

            result = result.detach().cpu().numpy()

            if not np.isfinite(result).all():
                raise ValueError(
                    f"Embeddings no finitos en la imagen {image_data['image_id']}"
                )

            embeddings[i] = result.astype(np.float16)

            embeddings.flush()
            done[i] = True

            if (i + 1) % 25 == 0 or i == n - 1:
                save_progress(
                    done, progress_path, metadata_path, n, signature
                )
                print(f"Progreso: {done.sum()}/{n}")

        embeddings.flush()
        save_progress(
            done, progress_path, metadata_path, n, signature
        )

    finally:
        del embeddings

    print(f"Embeddings completados: {done.sum()}/{n}")

def load_captions(captions_json_path):
    """
    Carga los captions de COCO y los agrupa por image_id.

    Returns:
        dict[int, list[str]]:
            Cada image_id tiene asociada una lista de captions.
    """
    with open(captions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    captions_by_image = defaultdict(list)

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        caption = annotation["caption"].strip()

        captions_by_image[image_id].append(caption)

    return dict(captions_by_image)

def save_captions_csv(images, output_dir):
    """
    Guarda las captions originales en un CSV para evaluar CLIP Score.

    Genera una fila por caption, con las columnas image_id y caption.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "captions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "caption"])

        for image_data in images:
            image_id = image_data["image_id"]

            for caption in image_data["captions"]:
                writer.writerow([image_id, caption])

    print(f"Captions guardadas en {csv_path}")

def main():
    annotation_file = Path(
        "coco/annotations/person_keypoints_train2014.json"
    )
    captions_file = Path(
        "coco/annotations/captions_train2014.json"
    )
    images_dir = Path("coco/train2014")

    device = "cpu"
    output_dir = Path("features/train")

    images = load_images(
        annotation_file=str(annotation_file),
        captions_file=str(captions_file),
        images_dir=str(images_dir),
        max_images=None
    )

    if not images:
        raise ValueError("No se encontraron imágenes válidas.")

    # Extraer o reanudar los latentes.
    vae = loadVAE(device)
    extract_dataset_latents(images, vae, output_dir, device)
    del vae

    # Extraer o reanudar los embeddings.
    tokenizer, text_model = load_clip(device)
    extract_dataset_text_embeddings(
        images, tokenizer, text_model, output_dir, device
    )
    del text_model

    print("Extracción completada.")


if __name__ == "__main__":
    main()