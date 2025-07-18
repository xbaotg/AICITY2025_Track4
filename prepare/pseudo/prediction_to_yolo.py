import json
import argparse
import os
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert prediction results to YOLO format.",
    )
    parser.add_argument(
        "--predictions", type=str, help="Path to the predictions JSON file"
    )
    parser.add_argument("--image_folder", type=str, help="Path to the image folder")
    parser.add_argument("--output_folder", type=str, help="Output folder")
    return parser.parse_args()


def convert_data(args):
    data = json.load(open(args.predictions, "r"))
    img_names = set()

    for item in data:
        img_names.add(item["image_id"])

    converted_data = {}
    for img_name in img_names:
        converted_data[img_name] = []

    for item in tqdm(data):
        img_name = item["image_id"]
        bbox = item["bbox"]
        category_id = item["category_id"]
        score = item["score"]

        converted_data[img_name].append(
            {
                "bbox": bbox,
                "category_id": category_id,
                "score": score,
            }
        )
    return converted_data


def save_batch(start, end, data, image_folder, output_folder):
    for img in tqdm(list(image_folder.glob("*"))[start:end]):
        img_stem = img.stem
        res = []
        for item in data.get(img_stem, []):
            bbox = item["bbox"]
            category_id = item["category_id"]
            score = item["score"]

            # Convert bbox to YOLO format
            width = bbox[2]
            height = bbox[3]
            x_center = bbox[0] + bbox[2] / 2
            y_center = bbox[1] + bbox[3] / 2

            # Normalize coordinates
            with Image.open(img) as img_obj:
                img_width, img_height = img_obj.size
                x_center /= img_width
                y_center /= img_height
                width /= img_width
                height /= img_height
            res.append(
                f"{int(category_id)} {x_center} {y_center} {width} {height} {score}\n"
            )

        if res:
            output_file = output_folder / f"{img_stem}.txt"
            with open(output_file, "w") as f:
                f.writelines(res)


def save_yolo_format(data, image_folder, output_folder):
    num_processes = os.cpu_count() or 16
    num_images = len(list(image_folder.glob("*")))
    batch_size = num_images // num_processes + 1
    batches = [
        (i, min(i + batch_size, num_images)) for i in range(0, num_images, batch_size)
    ]
    process_map(
        save_batch,
        [start for start, _ in batches],
        [end for _, end in batches],
        [data] * len(batches),
        [image_folder] * len(batches),
        [output_folder] * len(batches),
        max_workers=num_processes,
        desc="Saving YOLO format",
    )


def main():
    args = parse_args()
    if not Path(args.predictions).exists():
        print(f"File does not exist: {args.predictions}")
        exit(1)
    if not Path(args.image_folder).exists():
        print(f"Folder does not exist: {args.image_folder}")
        exit(1)

    if not Path(args.output_folder).exists():
        os.makedirs(args.output_folder, exist_ok=True)
    converted_data = convert_data(args)
    save_yolo_format(converted_data, Path(args.image_folder), Path(args.output_folder))


if __name__ == "__main__":
    main()
