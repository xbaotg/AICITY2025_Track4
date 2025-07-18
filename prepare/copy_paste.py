from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
import random
import math


def parse_args():
    parser = ArgumentParser(description="Copy-paste augmentation script")
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        nargs="+",
        help="Directory containing input images",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        nargs="+",
        help="Directory containing input annotations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to save augmented images and annotations",
    )
    parser.add_argument(
        "--max_generations",
        type=int,
        default=30,
        help="Maximum number of generations for augmentation",
    )
    return parser.parse_args()


def calculate_overlap(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x1 < x2 and y1 < y2:
        intersection_area = (x2 - x1) * (y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - intersection_area
        return intersection_area / union_area

    return 0.0


def group_images_by_camera_id(data):
    unique_camera_ids = set(["_".join(image.name.split("_")[:2]) for image in data])
    grouped_images = {camera_id: [] for camera_id in unique_camera_ids}

    for camera_id in unique_camera_ids:
        for image in data:
            if image.name.startswith(camera_id):
                grouped_images[camera_id].append(image)

    return grouped_images


def get_annotations_for_image(image, annotations):
    return [annotation for annotation in annotations if annotation.stem == image.stem]


def load_annotation(image_path, annotation_path):
    image = Image.open(image_path)
    with open(annotation_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    annotations = []
    for line in lines:
        parts = line.strip().split()
        cls_id, x_center, y_center, width, height = map(float, parts[:5])
        x_center, y_center, width, height = (
            x_center * image.width,
            y_center * image.height,
            width * image.width,
            height * image.height,
        )

        annotations.append(
            {
                "class_id": int(cls_id),
                "bbox": (
                    x_center - width / 2,
                    y_center - height / 2,
                    x_center + width / 2,
                    y_center + height / 2,
                ),
                "cropped_image": image.crop(
                    (
                        x_center - width / 2,
                        y_center - height / 2,
                        x_center + width / 2,
                        y_center + height / 2,
                    )
                ),
            }
        )

    return annotations


def draw_comparison(image1, image2, annotations1, annotations2):
    # Draw two images side by side with their annotations using Pillow
    from PIL import ImageDraw

    # Create a new image to combine both images side by side
    combined_width = image1.width + image2.width
    combined_height = max(image1.height, image2.height)
    combined_image = Image.new("RGB", (combined_width, combined_height))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))

    # Draw annotations on the first image
    draw1 = ImageDraw.Draw(combined_image)
    for annotation in annotations1:
        bbox = annotation["bbox"]
        draw1.rectangle(
            [bbox[0], bbox[1], bbox[2], bbox[3]],
            outline="red",
            width=2,
        )

    # Draw annotations on the second image
    draw2 = ImageDraw.Draw(combined_image)
    for annotation in annotations2:
        bbox = annotation["bbox"]
        draw2.rectangle(
            [bbox[0] + image1.width, bbox[1], bbox[2] + image1.width, bbox[3]],
            outline="blue",
            width=2,
        )

    # Save the combined image
    combined_image.save("comparison.png")


def save_annotations(image_path, image, annotations, output_dir):
    annotation_file = output_dir / "labels" / f"{image_path.stem}.txt"
    annotation_file.parent.mkdir(parents=True, exist_ok=True)

    with open(annotation_file, "w", encoding="utf-8") as file:
        for annotation in annotations:
            cls_id = annotation["class_id"]
            x_center = (annotation["bbox"][0] + annotation["bbox"][2]) / 2 / image.width
            y_center = (
                (annotation["bbox"][1] + annotation["bbox"][3]) / 2 / image.height
            )
            width = (annotation["bbox"][2] - annotation["bbox"][0]) / image.width
            height = (annotation["bbox"][3] - annotation["bbox"][1]) / image.height
            file.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
            
    output_image_path = output_dir / "images" / image_path.name
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_image_path)


def augment_copy_paste(images, annotation_paths, max_generations=10, output_dir=None):
    grouped_images = group_images_by_camera_id(images)
    results = []

    print("Initial number of images:", len(images))
    mapping_annotations = {}
    for image in tqdm(images):
        annotation_file = get_annotations_for_image(image, annotation_paths)
        if annotation_file:
            mapping_annotations[image.name] = annotation_file[0]

    for camera_id, images in tqdm(grouped_images.items(), desc="Processing cameras"):
        annotations = []

        for image in images:
            if image.name in mapping_annotations:
                annotations.extend(load_annotation(image, mapping_annotations[image.name]))

        print("Grouped images for camera ID:", camera_id)
        print("Number of images:", len(images))
        print("Number of annotations:", len(annotations))

        for idx, image in enumerate(images):
            if image.name not in mapping_annotations:
                continue
                
            original_image = Image.open(image)
            augmented_image = original_image.copy()

            random.seed(idx)
            random.shuffle(annotations)

            annotation_image = load_annotation(image, mapping_annotations[image.name])
            cnt = 0
            
            image_center_x = original_image.width / 2
            image_center_y = original_image.height / 2
            max_distance = math.sqrt(image_center_x**2 + image_center_y**2)

            for annotation_of_group in annotations:
                if cnt >= max_generations:
                    break

                bbox = annotation_of_group["bbox"]
                bbox_center_x = (bbox[0] + bbox[2]) / 2
                bbox_center_y = (bbox[1] + bbox[3]) / 2
                distance = math.sqrt(
                    (bbox_center_x - image_center_x) ** 2
                    + (bbox_center_y - image_center_y) ** 2
                )
                probability = distance / max_distance + 0.3

                if random.random() > probability:
                    continue

                should_add = True

                for annotation_of_image in annotation_image:
                    iou_overlap = calculate_overlap(
                        annotation_of_group["bbox"], annotation_of_image["bbox"]
                    )

                    if iou_overlap > 0.1:
                        should_add = False
                        break

                if should_add:
                    annotation_image.append(annotation_of_group)
                    bbox = tuple(map(int, annotation_of_group["bbox"]))
                    cropped_image = annotation_of_group["cropped_image"]
                    augmented_image.paste(
                        cropped_image,
                        (bbox[0], bbox[1]),
                    )
                    cnt += 1

            save_annotations(image, augmented_image, annotation_image, output_dir)

    return results


def main():
    args = parse_args()

    images = args.images
    annotations = args.annotations
    output_dir = args.output
    max_generations = args.max_generations

    output_dir.mkdir(parents=True, exist_ok=True)

    images = [list(Path(image).glob("*")) for image in images]
    images = [image for sublist in images for image in sublist]

    annotations = [list(Path(annotation).glob("*.txt")) for annotation in annotations]
    annotations = [annotation for sublist in annotations for annotation in sublist]

    augment_copy_paste(images, annotations, max_generations, output_dir)


if __name__ == "__main__":
    main()
