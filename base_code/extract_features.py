import json
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from torchvision import transforms


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
    # Load person keypoints annotations.
    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load captions annotations.
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

    # Keep images with person bounding boxes and five captions.
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

        # Stop once the requested number of images is reached.
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

    # 2. Add a margin around the union.
    bbox = expand_bbox(bbox, margin)

    # 3. Make the bounding box square.
    bbox = make_square(bbox)

    # 4. Adjust the bounding box to the image boundaries.
    bbox = clip_bbox(bbox, image_width, image_height)

    # Convert coordinates to integers.
    x_min, y_min, width, height = bbox

    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_min + width)
    y_max = int(y_min + height)

    # 5. Crop and resize the image.
    cropped_image = image.crop((x_min, y_min, x_max, y_max))
    cropped_image = cropped_image.resize(
        (output_size, output_size),
        Image.Resampling.BICUBIC
    )

    # 6. Convert to tensor and normalize to [-1, 1].
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


if __name__ == "__main__":

    annotation_file = Path(
        "coco/annotations/person_keypoints_train2014.json"
    )
    captions_file = Path(
        "coco/annotations/captions_train2014.json"
    )
    images_dir = Path("coco/train2014")

    # Check that the paths exist.
    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {annotation_file}"
        )

    if not captions_file.exists():
        raise FileNotFoundError(
            f"Captions file not found: {captions_file}"
        )

    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}"
        )

    # Load metadata for just one image.
    images = load_images(
        annotation_file=str(annotation_file),
        captions_file=str(captions_file),
        images_dir=str(images_dir),
        max_images=1
    )

    if not images:
        raise ValueError(
            "No images with person bounding boxes and five captions "
            "were found."
        )

    # Select the first image.
    image_data = images[0]

    print("Image ID:", image_data["image_id"])
    print("Image path:", image_data["image_path"])
    print("Bounding boxes:", image_data["bboxes"])

    # Display the five captions.
    print("\nCaptions:")
    for i, caption in enumerate(image_data["captions"], start=1):
        print(f"  {i}. {caption}")

    # Check that the image has exactly five captions.
    assert len(image_data["captions"]) == 5

    # Display the original image and its preprocessed version.
    test_preprocess_image(
        image_path=str(image_data["image_path"]),
        bboxes=image_data["bboxes"]
    )
