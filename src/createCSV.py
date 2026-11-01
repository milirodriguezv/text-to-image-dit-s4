from extract_features import *

def main():
    annotation_file = Path("coco/annotations/person_keypoints_train2014.json")
    captions_file = Path("coco/annotations/captions_train2014.json")
    images_dir = Path("coco/train2014")
    output_dir = Path("features/train")

    # Cargar los metadatos de las imágenes y sus captions.
    # None permite procesar todas las imágenes válidas.
    images = load_images(
        annotation_file=str(annotation_file),
        captions_file=str(captions_file),
        images_dir=str(images_dir),
        max_images=None
    )

    if not images:
        raise ValueError("No se encontraron imágenes válidas.")

    # Guardar las captions para la evaluación posterior con CLIP Score.
    save_captions_csv(
        images=images,
        output_dir=output_dir
    )

    print(f"Proceso terminado. Imágenes procesadas: {len(images)}")


if __name__ == "__main__":
    main()
