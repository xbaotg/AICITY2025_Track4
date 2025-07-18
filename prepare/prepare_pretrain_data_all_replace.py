import shutil
import json
import random
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from COCO import ALL_COCO_CLASSES
from PIL import Image
import multiprocessing
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

CLASSES = {0: "bus", 1: "bike", 2: "car", 3: "pedestrian", 4: "truck"}
MAPPING_TO_COCO = {"bus": 5, "bike": 3, "car": 2, "pedestrian": 0, "truck": 7}

MAPPING_VISDRONE_TO_FISHEYE = {
    # perdestrian : perdestrian
    0: 3,
    # people : perdestrian
    # 1: 3,
    # bycicle : bike
    2: 1,
    # car : car
    3: 2,
    # van : car
    4: 2,
    # truck : truck
    5: 4,
    # # tricycle : bike
    # 6: 1,
    # # awning-trycycle : bike
    # 7: 1,
    # bus : bus
    8: 0,
    # motor : bike
    9: 1,
}


def parse_args():
    parser = ArgumentParser(description="Prepare original data for training.")
    parser.add_argument(
        "--images", type=str, nargs="+", help="Directories containing image files."
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", help="Directories containing label files."
    )
    parser.add_argument(
        "--output", type=str, help="Directory to save COCO annotations.", required=True
    )

    parser.add_argument(
        "--additional_images",
        type=str,
        nargs="+",
        help="Directories containing additional image files.",
    )
    parser.add_argument(
        "--additional_labels",
        type=str,
        nargs="+",
        help="Directories containing additional label files.",
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["yolo", "dfine", "rfdetr"],
        default="yolo",
        help="Type of annotations to process.",
    )
    return parser.parse_args()


def split_dataset(items, ratios):
    random.seed(42)
    random.shuffle(items)
    total = len(items)
    train_size = int(ratios[0] * total)
    val_size = int(ratios[1] * total)
    train = items[:train_size]
    val = items[train_size : train_size + val_size]
    test = items[train_size + val_size :]
    return train, val, test


def group_by_pattern(images, patterns):
    grouped = defaultdict(list)
    for image in images:
        for pattern in patterns:
            if pattern in image.name:
                grouped[pattern].append(image)
                break
    return grouped


def create_rfdetr_annotations(image_paths, label_paths, output_file, images_output_dir):
    annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": coco_id, "name": name, "supercategory": "none"}
            # for coco_id, name in ALL_COCO_CLASSES.items()
            for coco_id, name in CLASSES.items()
        ],
    }
    annotation_id = 1
    for idx, image_path in tqdm(enumerate(image_paths)):
        image_id = idx + 1
        new_image_path = images_output_dir / image_path.name
        shutil.copy(image_path, new_image_path)  # Uncommenting the copy operation

        # Open the image to get width and height
        with Image.open(image_path) as img:
            width, height = img.size

        annotations["images"].append(
            {
                "id": image_id,
                "file_name": new_image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_file = label_paths.get(image_path.stem)
        if label_file and label_file.exists():
            with open(label_file, "r") as lf:
                for line in lf:
                    class_id, x_center_norm, y_center_norm, width_norm, height_norm = (
                        map(float, line.strip().split())
                    )
                    # coco_id = MAPPING_TO_COCO[CLASSES[int(class_id)]]
                    coco_id = int(class_id)

                    # Convert normalized YOLO values to absolute pixel values
                    x_center = x_center_norm * width
                    y_center = y_center_norm * height
                    bbox_width = width_norm * width
                    bbox_height = height_norm * height
                    x_min = x_center - bbox_width / 2
                    y_min = y_center - bbox_height / 2

                    annotations["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": coco_id,
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1

    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=4)


def create_dfine_annotations(image_paths, label_paths, output_file, images_output_dir):
    annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": coco_id, "name": name} for coco_id, name in CLASSES.items()
        ],
    }
    annotation_id = 1
    print(len(image_paths))
    bar = tqdm(total=len(image_paths), desc="Processing images")
    for idx, image_path in enumerate(image_paths):
        image_id = idx + 1
        new_image_path = images_output_dir / image_path.name
        shutil.copy(image_path, new_image_path)

        # Open the image to get width and height
        with Image.open(image_path) as img:
            width, height = img.size

        annotations["images"].append(
            {
                "id": image_id,
                "file_name": new_image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_file = label_paths.get(image_path.stem)
        if label_file and label_file.exists():
            with open(label_file, "r") as lf:
                for line in lf:
                    z = line.strip().split()

                    if len(z) == 6:
                        z = z[:-1]

                    (
                        class_id,
                        x_center_norm,
                        y_center_norm,
                        width_norm,
                        height_norm,
                    ) = map(float, z)

                    coco_id = int(class_id)

                    # Convert normalized YOLO values to absolute pixel values
                    x_center = x_center_norm * width
                    y_center = y_center_norm * height
                    bbox_width = width_norm * width
                    bbox_height = height_norm * height
                    x_min = x_center - bbox_width / 2
                    y_min = y_center - bbox_height / 2

                    annotations["annotations"].append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": coco_id,
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "area": bbox_width * bbox_height,
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1

        bar.update(1)

    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=4)


def create_yolo_annotations(image_paths, label_paths, output_path, label_output_path):
    output_path.mkdir(parents=True, exist_ok=True)
    label_output_path.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_paths):
        image_id = image_path.stem

        new_image_path = output_path / image_path.name
        shutil.copy(image_path, new_image_path)  # Uncommenting the copy operation

        label_file = label_paths.get(image_id)
        if label_file and label_file.exists():
            with open(label_file, "r") as lf:
                lines = lf.readlines()
                with open(label_output_path / f"{image_path.stem}.txt", "w") as of:
                    for line in lines:
                        (
                            class_id,
                            x_center_norm,
                            y_center_norm,
                            width_norm,
                            height_norm,
                        ) = map(float, line.strip().split())

                        of.write(
                            f"{int(class_id)} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"
                        )
        else:
            print(f"Label file for {image_path.name} not found.")


def main(
    images_dirs,
    labels_dirs,
    output_dir,
    output_type,
    additional_images=None,
    additional_labels=None,
):
    # patterns = ["_A_", "_M_", "_E_", "_N_"]
    # images = [Path(image).glob("*.png") for image in images_dirs]
    # images = [img for sublist in images for img in sublist]
    # labels = [Path(label).glob("*.txt") for label in labels_dirs]
    # labels = {lbl.stem: lbl for sublist in labels for lbl in sublist}

    # grouped_images = group_by_pattern(images, patterns)
    # train_set, val_set, test_set = [], [], []

    # for pattern, group in grouped_images.items():
    #     train, val, test = split_dataset(group, (0.9, 0.1, 0.0))
    #     train_set.extend(train)
    #     val_set.extend(val)
    #     test_set.extend(test)
    images = [Path(image).glob("*") for image in images_dirs]
    images = [img for sublist in images for img in sublist]
    labels = [Path(label).glob("*.txt") for label in labels_dirs]
    labels = {lbl.stem: lbl for sublist in labels for lbl in sublist}

    if additional_labels:
        for additional_label in additional_labels:
            additional_labels_list = list(Path(additional_label).glob("*.txt"))
            labels.update({lbl.stem: lbl for lbl in additional_labels_list})
            print(f"Added {len(additional_labels_list)} labels from {additional_label}")
            
    if additional_images:
        for additional_image in additional_images:
            additional_images_list = list(Path(additional_image).glob("*"))
            additional_images_list = [
                img for img in additional_images_list if img.suffix.lower() in [".jpg", ".png"]
            ]

            # replace in images with same name
            temp_stem = [img.stem for img in images]
            for img in additional_images_list:
                if img.stem in temp_stem:
                    idx = temp_stem.index(img.stem)
                    print("Replacing image:", images[idx], "with", img)
                    images[idx] = img
                else:
                    images.append(img)

    # spliting the dataset into train, validation, and test sets
    train_set, val_set, test_set = [], [], []
    train, val, test = split_dataset(images, (0.9, 0.1, 0.0))
    train_set.extend(train)
    val_set.extend(val)
    test_set.extend(test)

    print("Number of training images:", len(train_set))
    print("Number of validation images:", len(val_set))
    print("Number of testing images:", len(test_set))
    print("Number of label files:", len(list(labels.keys())))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if output_type == "rfdetr":
        train_output_dir = output_path / "train"
        train_output_dir.mkdir(parents=True, exist_ok=True)

        val_output_dir = output_path / "valid"
        val_output_dir.mkdir(parents=True, exist_ok=True)

        test_output_dir = output_path / "test"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        # multiprocess the create_coco_annotations function

        process_map(
            create_rfdetr_annotations,
            [train_set, val_set, test_set],
            [labels, labels, labels],
            [
                train_output_dir / "_annotations.coco.json",
                val_output_dir / "_annotations.coco.json",
                test_output_dir / "_annotations.coco.json",
            ],
            [train_output_dir, val_output_dir, test_output_dir],
            max_workers=4,
        )

    elif output_type == "yolo":
        train_output_dir = output_path / "images" / "train"
        tarin_label_dir = output_path / "labels" / "train"

        val_output_dir = output_path / "images" / "valid"
        val_label_dir = output_path / "labels" / "valid"

        test_output_dir = output_path / "images" / "test"
        test_label_dir = output_path / "labels" / "test"

        process_map(
            create_yolo_annotations,
            [train_set, val_set, test_set],
            [labels, labels, labels],
            [
                train_output_dir,
                val_output_dir,
                test_output_dir,
            ],
            [
                tarin_label_dir,
                val_label_dir,
                test_label_dir,
            ],
        )

    elif output_type == "dfine":
        images_output_dir = output_path / "images"
        images_output_dir.mkdir(parents=True, exist_ok=True)

        process_map(
            create_dfine_annotations,
            [train_set, val_set, test_set],
            [labels, labels, labels],
            [
                output_path / "train.json",
                output_path / "val.json",
                output_path / "test.json",
            ],
            [images_output_dir, images_output_dir, images_output_dir],
        )

        # create_coco_annotations(
        #     train_set, labels, output_path / "train.json", images_output_dir
        # )
        # create_coco_annotations(
        #     val_set, labels, output_path / "val.json", images_output_dir
        # )
        # create_coco_annotations(
        #     test_set, labels, output_path / "test.json", images_output_dir
        # )

    # create_coco_annotations(
    #     train_set,
    #     labels,
    #     train_output_dir / "_annotations.coco.json",
    #     train_output_dir,
    # )

    # create_coco_annotations(
    #     val_set, labels, val_output_dir / "_annotations.coco.json", val_output_dir
    # )

    # create_coco_annotations(
    #     test_set, labels, test_output_dir / "_annotations.coco.json", test_output_dir
    # )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.images,
        args.labels,
        args.output,
        args.type,
        args.additional_images,
        args.additional_labels,
    )

# python prepare_pretrain_data_all.py --images ../collections/unlabeled/aip_vip_fisheye/images/ --labels ../collections/unlabeled/aip_vip_fisheye/labels/ --output ../collections/prepared/pseudo_\(aip_vip_fisheye\)_fe8k_dfine --type dfine --additional_images ../collections/raw/fisheye8k/train/images/ ../collections/raw/fisheye8k/test/images/ --additional_labels ../collections/raw/fisheye8k/train/labels/ ../collections/raw/fisheye8k/test/labels/
