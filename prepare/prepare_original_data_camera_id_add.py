import shutil
import json
import random
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
from COCO import ALL_COCO_CLASSES
from PIL import Image

CLASSES = {0: "bus", 1: "bike", 2: "car", 3: "pedestrian", 4: "truck"}
# MAPPING_TO_COCO = {"bus": 5, "bike": 3, "car": 2, "pedestrian": 0, "truck": 7}
print("Using classes:", CLASSES)

train_camera_id = [1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15]
test_camera_id = [6, 18, 16, 12, 17, 5]


def parse_args():
    parser = ArgumentParser(description="Prepare original data for training.")
    parser.add_argument(
        "--images", type=str, nargs="+", help="Directories containing image files."
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", help="Directories containing label files."
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
        "--output", type=str, help="Directory to save COCO annotations.", required=True
    )
    return parser.parse_args()


def create_coco_annotations(image_paths, label_paths, output_file, images_output_dir):
    annotations = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": coco_id, "name": name} for coco_id, name in CLASSES.items()
        ],
    }

    annotation_id = 1
    for idx, image_path in tqdm(enumerate(image_paths)):
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
                    if len(line.strip().split(" ")) == 6:
                        line = line.strip().split()[:5]  # Handle 5-column YOLO format
                    else:
                        line = line.strip().split()
                    class_id, x_center_norm, y_center_norm, width_norm, height_norm = (
                        map(float, line)
                    )
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


def main(
    images_dirs, labels_dirs, output_dir, additional_images=None, additional_labels=None
):
    train_ratio = 0.8
    train_pattern = [f"camera{camera_id}_" for camera_id in train_camera_id]
    test_pattern = [f"camera{camera_id}_" for camera_id in test_camera_id]

    # hardcoded patterns for train and test sets

    train_images = []
    test_images = []

    for pattern in train_pattern:
        images = [list(Path(image).glob(f"{pattern}*.png")) for image in images_dirs]
        train_images.extend([img for sublist in images for img in sublist])

    for pattern in test_pattern:
        images = [list(Path(image).glob(f"{pattern}*.png")) for image in images_dirs]
        test_images.extend([img for sublist in images for img in sublist])

    # camera4_E
    camera_4_e = "camera4_E_"
    images_camera_4_e = [
        list(Path(image).glob(f"{camera_4_e}*.png")) for image in images_dirs
    ]
    train_images_camera_4_e = [
        sublist[: int(len(sublist) * train_ratio)] for sublist in images_camera_4_e
    ]
    test_images_camera_4_e = [
        sublist[int(len(sublist) * train_ratio) :] for sublist in images_camera_4_e
    ]
    train_images.extend([img for sublist in train_images_camera_4_e for img in sublist])
    test_images.extend([img for sublist in test_images_camera_4_e for img in sublist])

    # camera4_A/M/N
    patterns_camera_4 = ["camera4_A_", "camera4_M_", "camera4_N_"]
    images_camera_4 = [
        list(Path(image).glob(f"{pattern}*.png"))
        for image in images_dirs
        for pattern in patterns_camera_4
    ]
    test_images_camera_4 = [sublist for sublist in images_camera_4]
    test_images.extend([img for sublist in test_images_camera_4 for img in sublist])

    if additional_images:
        for additional_image in additional_images:
            additional_images_list = list(Path(additional_image).glob("*"))
            train_images.extend(additional_images_list)
            print(f"Added {len(additional_images_list)} images from {additional_image}")

    print("Number of training images:", len(train_images))
    print("Number of testing images:", len(test_images))

    # patterns = ["_A_", "_M_", "_E_", "_N_"]
    # images = [Path(image).glob("*.png") for image in images_dirs]
    # images = [img for sublist in images for img in sublist]
    labels = [Path(label).glob("*.txt") for label in labels_dirs] + [
        Path(label).glob("*.txt") for label in additional_labels
    ]
    labels = {lbl.stem: lbl for sublist in labels for lbl in sublist}
    print("Number of label files:", len(list(labels.keys())))

    # grouped_images = group_by_pattern(images, patterns)
    # train_set, val_set, test_set = [], [], []

    # for pattern, group in grouped_images.items():
    #     train, val, test = split_dataset(group, (0.8, 0.1, 0.1))
    #     train_set.extend(train)
    #     val_set.extend(val)
    #     test_set.extend(test)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_output_dir = output_path / "images"
    images_output_dir.mkdir(parents=True, exist_ok=True)

    create_coco_annotations(
        train_images, labels, output_path / "train.json", images_output_dir
    )
    create_coco_annotations(
        test_images, labels, output_path / "val.json", images_output_dir
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.images,
        args.labels,
        args.output,
        args.additional_images,
        args.additional_labels,
    )

# python prepare_original_data_camera_id_add.py --images ..//collections/raw/fisheye8k/train/images/ ../collections/raw/fisheye8k/test/images/ --labels ../collections/raw/fisheye8k/test/labels/ ../collections/raw/fisheye8k/train/labels/ --output ../collections/prepared/updated_fisheye_w_visdrone --additional_images ../collections/raw/visdrone2019/all_yolo/images/ --additional_labels ../collections/raw/visdrone2019/all_yolo/labels/
# python prepare_original_data_camera_id_add.py --images ../collections/raw/fisheye8k/train/images/ ../collections/raw/fisheye8k/test/images/ --labels ../collections/unlabeled/aip_vip_fisheye/labels/ ../collections/unlabeled/aip_vip_fisheye/labels/ --output ../collections/prepared/updated_fisheye_w_aip_vip_fsk --additional_images ../collections/unlabeled/aip_vip_fisheye/images/ --additional_labels ../collections/unlabeled/aip_vip_fisheye/labels/
