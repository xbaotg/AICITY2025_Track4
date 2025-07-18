from pathlib import Path
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare unlabeled data for pseudo labeling.",
    )
    parser.add_argument("--image_folder", help="List of image folders", nargs="+")
    parser.add_argument("--output_folder", type=str, help="Output folder")
    parser.add_argument(
        "--p_num", type=int, default=8, help="Number of processes to use"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite the output folder if it exists",
    )
    return parser.parse_args()


def _copy(image_paths, output_folder, start, end, force=False):
    """
    Copy a single image to the output folder.
    Args:
        image_path (Path): Path to the image file.
        output_folder (Path): Path to the output folder.
    """
    for image_path in tqdm(image_paths[start:end]):
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"Image {image_path} does not exist.")
            continue
        new_image_path = output_folder / image_path.name
        if new_image_path.exists() and not force:
            continue
        shutil.copy(image_path, new_image_path)


def copy_images(image_folders, output_folder, p_num, force=False):
    """
    Copy images from multiple folders to a single output folder.
    Args:
        image_folders (list): List of folders containing images.
        output_folder (str): Path to the output folder.
        p_num (int): Number of processes to use.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    all_image_paths = []

    extensions = ["jpg", "jpeg", "png"]

    for image_folder in image_folders:
        image_folder = Path(image_folder)
        for ext in extensions:
            for image_path in image_folder.glob(f"*.{ext}"):
                all_image_paths.append(image_path)
    print(f"Found {len(all_image_paths)} images in {len(image_folders)} folders.")
    all_image_paths = sorted(all_image_paths)

    image_paths_per_process = len(all_image_paths) // p_num

    starts = []
    ends = []
    for i in range(p_num):
        start = i * image_paths_per_process
        end = (
            (i + 1) * image_paths_per_process
            if (i + 1) * image_paths_per_process < len(all_image_paths)
            else len(all_image_paths)
        )
        starts.append(start)
        ends.append(end)

    process_map(
        _copy,
        [all_image_paths] * p_num,
        [output_folder] * p_num,
        starts,
        ends,
        [force] * p_num,
    )

    print(f"Copied {len(all_image_paths)} images to {output_folder}.")
    return all_image_paths


def main():
    args = parse_args()
    if not Path(args.output_folder).exists():
        Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    all_image_paths = copy_images(
        args.image_folder, args.output_folder, args.p_num, args.force
    )
    print(f"Copied {len(all_image_paths)} images to {args.output_folder}.")


if __name__ == "__main__":
    main()

# python pseudo/prepare_unlabeled_data.py --image_folder ../collections/raw/VIPCUP2020/aip-cup-2020/vipcup2020/vipcup2020/images/train/ ../collections/raw/VIPCUP2020/aip-cup-2020/vipcup2020/vipcup2020/images/val/ ../collections/raw/VIPCUP2020/vip-cup-2020/vip_cup_2020/fisheye-day-30062020/images/train/ ../collections/raw/fisheye8k/test/images/ ../collections/raw/fisheye8k/train/images/
