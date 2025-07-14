import argparse
import os
import torch
import time
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import bbox_visualizer as bbv
import pyspng, time


from eval import evaluate


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Model inference")

    # Common arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "dfine-conf-dis",
            "dfinetrt-conf",
        ],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input folder containing images",
    )
    parser.add_argument(
        "--output", type=str, default="output", help="Path to output folder"
    )
    parser.add_argument(
        "--measure-flops", action="store_true", help="Measure model FLOPs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize predictions"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate model performance"
    )
    # Add --fast argument
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use DataLoader for faster evaluation (batch processing)",
    )

    # Model-specific arguments
    dfine_group = parser.add_argument_group("D-FINE model arguments")
    dfine_group.add_argument(
        "--checkpoint", type=str, help="Path to D-FINE model checkpoint"
    )
    dfine_group.add_argument("--config", type=str, help="Path to D-FINE model config")
    dfine_group.add_argument("--input-size", type=int, default=640)

    # Eval arguments
    parser.add_argument(
        "--gt",
        type=str,
        help="Path to the ground truth annotations.",
    )
    parser.add_argument(
        "--type",
        type=int,
        choices=[5, 80],
        help="Type of evaluation: 5 for AICity, 80 for COCO.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show inference log or not",
    )

    args = parser.parse_args()

    if args.eval:
        if args.gt is None:
            parser.error("--gt is required for evaluation")

        if args.type is None:
            parser.error("--type is required for evaluation")
    return args


def load_model(args):
    """Load the appropriate model based on the specified model type."""
    if args.model == "dfinetrt-custom":
        from models.dfine_trt_custom import DFineTRTModel

        print(
            f"Loading D-FINE TRT Custom model from {args.checkpoint} with config {args.config}"
        )
        model = DFineTRTModel()
        model.load_model(args.checkpoint, args.input_size, args.type)

        return model
    elif args.model == "dfine-conf-dis":
        from models.dfine_custom_dis import DFineModel

        print(
            f"Loading D-FINE Custom model from {args.checkpoint} with config {args.config}"
        )
        model = DFineModel()
        model.load_model(args.checkpoint, args.config, args.input_size, args.type)

        return model
    else:
        raise ValueError(f"Unsupported model type: {args.model}")


def get_image_paths(input_dir):
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(Path(input_dir).rglob(f"*{ext}"))

    return image_paths


def visualize_results(image, results, output_path):
    colors_labels = {
        "bus": (255, 99, 71),  # Tomato Red
        "bike": (135, 206, 250),  # Sky Blue
        "car": (50, 205, 50),  # Lime Green
        "pedestrian": (218, 112, 214),  # Orchid
        "truck": (255, 165, 0),  # Orange
    }

    labels = [res["cls"] for res in results]
    scores = [res["conf"] for res in results]
    w, h = image.shape[:2]
    for i, res in enumerate(results):
        res["bbox"] = [
            int(res["bbox"][0]),
            int(res["bbox"][1]),
            int(res["bbox"][2]),
            int(res["bbox"][3]),
        ]

    bboxes = [res["bbox"] for res in results]

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for i, label in enumerate(labels):
        if bboxes[i][2] <= bboxes[i][0] or bboxes[i][3] <= bboxes[i][1]:
            continue
        color = colors_labels.get(label, (255, 255, 255))
        image = bbv.draw_rectangle(image, bboxes[i], bbox_color=color, thickness=2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image)


def main():
    # Parse arguments
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Initialize model
    model = load_model(args)

    # Get image paths
    image_paths = get_image_paths(args.input)
    print(f"Found {len(image_paths)} images for inference")

    # Initialize timing variables
    total_time = 0
    total_images = 0
    total_load_time = 0
    all_results = {}

    # Warm up the model
    warmup_images = [image_paths[0] for _ in range(30)]
    for img_path in warmup_images:
        with open(str(img_path), "rb") as fin:
            image = pyspng.load(fin.read())

        model.inference(image, conf_thres=args.conf_thresh)

    # import pyvips
    def padding(img: np.ndarray) -> np.ndarray:
        """
        paddding the image to square shape with the input size.
        """
        h, w, _ = img.shape
        if h == w:
            return img

        if h > w:
            pad = (h - w) // 2
            img = np.pad(img, ((0, 0), (pad, pad), (0, 0)), mode="constant")
        else:
            pad = (w - h) // 2
            img = np.pad(img, ((pad, pad), (0, 0), (0, 0)), mode="constant")

        # cv2.imwrite("padding.jpg", img)
        # input()

        return img

    def de_padding_box(boxes: np.ndarray, orig_size: np.ndarray) -> np.ndarray:
        """
        Remove padding from the bounding boxes.
        """
        h, w, _ = orig_size.shape

        if h == w:
            return boxes

        if h > w:
            pad = (h - w) // 2
            boxes[0] -= pad
            boxes[2] -= pad

        else:
            pad = (w - h) // 2
            boxes[1] -= pad
            boxes[3] -= pad

        # boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        # boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        # boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
        # boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
        return boxes

    if args.fast:
        from torch.utils.data import Dataset, DataLoader

        class InferenceDataset(Dataset):
            def __init__(self, image_paths):
                self.image_paths = image_paths

            def __len__(self):
                return len(self.image_paths)

            def __getitem__(self, idx):
                path = self.image_paths[idx]
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                return image, str(path)

        def collate_fn_custom(batch):
            return batch[0]

        dataset = InferenceDataset(image_paths)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn_custom,
            pin_memory=True,
        )

        processing_start_time = time.time()

        for orig_image, img_path_str in tqdm(dataloader):
            img_path = Path(img_path_str)
            is_night = False

            if "_E_" in str(img_path) or "_N_" in str(img_path):
                is_night = True

            # image = padding(orig_image[0].clone().numpy())
            # print(image.shape)
            start_time = time.time()
            if args.model == "dfine-conf-dn":
                results = model.inference(
                    orig_image,
                    conf_thres=args.conf_thresh,
                    filename=str(img_path.stem),
                )
            else:
                results = model.inference(
                    orig_image, conf_thres=args.conf_thresh, is_night=is_night
                )

            inference_time = time.time() - start_time

            if args.verbose:
                print("time to inference: ", inference_time * 1000)
            all_results[str(img_path.name)] = results

            total_time += inference_time
            total_images += 1

            img_filename = os.path.basename(img_path)
            base_name = os.path.splitext(img_filename)[0]

            if args.verbose:
                print(
                    f"Image: {img_filename}, Detections: {len(results)}, Inference time: {inference_time * 1000:.2f}ms"
                )

            if args.visualize:
                vis_output_path = os.path.join(
                    args.output, "visualizations", f"{base_name}_vis.jpg"
                )
                os.makedirs(os.path.dirname(vis_output_path), exist_ok=True)
                visualize_results(orig_image, results, vis_output_path)

        total_processing_time = time.time() - processing_start_time
        total_load_time = total_processing_time - total_time
    else:

        def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
            """
            Read an image from a file.

            Args:
                filename (str): Path to the file to read.
                flags (int, optional): Flag that can take values of cv2.IMREAD_*. Defaults to cv2.IMREAD_COLOR.

            Returns:
                (np.ndarray): The read image.
            """
            return cv2.imdecode(np.fromfile(filename, np.uint8), flags)

        # Process each image
        for img_path in tqdm(image_paths):
            # Read image

            # Measure inference time

            # image = Image.open(img_path)
            # with img_path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            #     image = pyspng.load(mm)

            # with open(str(img_path), "rb") as fin:
            # image = pyspng.load(fin.read())
            start_time = time.time()
            # image = pyspng.load(open(str(img_path), "rb").read())
            # image = imread(str(img_path))
            orig_image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            # image = padding(orig_image.copy())
            image = orig_image.copy()
            load_time = time.time() - start_time
            is_night = False

            if "_E_" in str(img_path) or "_N_" in str(img_path):
                is_night = True
            # image = pyvips.Image.new_from_file(str(img_path), access="sequential").numpy()

            if args.verbose:
                print("time to load image: ", load_time * 1000)

            # image = pyvips.Image.new_from_file(str(img_path), access="sequential").numpy()
            if args.model == "dfine-conf-dn":
                start_time = time.time()
                results = model.inference(
                    image,
                    conf_thres=args.conf_thresh,
                    filename=str(img_path.stem),
                    is_night=is_night,
                )

                inference_time = time.time() - start_time
            else:
                start_time = time.time()
                results = model.inference(
                    image, conf_thres=args.conf_thresh, is_night=is_night
                )
                # forres in results:
                #     res["bbox"] = de_padding_box(res["bbox"], orig_image)

                inference_time = time.time() - start_time
            if args.verbose:
                print("time to inference: ", inference_time * 1000)
            all_results[str(img_path.name)] = results

            # Update timing stats
            total_time += inference_time
            total_load_time += load_time
            total_images += 1

            # Create output path for JSON results
            img_filename = os.path.basename(img_path)
            base_name = os.path.splitext(img_filename)[0]

            # Print results
            if args.verbose:
                print(
                    f"Image: {img_filename}, Detections: {len(results)}, Inference time: {inference_time * 1000:.2f}ms"
                )

            # Visualize if requested
            if args.visualize:
                vis_output_path = os.path.join(
                    args.output, "visualizations", f"{base_name}_vis.jpg"
                )
                os.makedirs(os.path.dirname(vis_output_path), exist_ok=True)
                visualize_results(orig_image, results, vis_output_path)

    # if args.model == "rfdetrtrt":
    #     model.context.pop()

    # Print summary
    if total_images > 0:
        avg_time = total_time / total_images
        print(f"\nProcessed {total_images} images")
        print(f"\nNum predictions: {sum(len(v) for v in all_results.values())}")
        print(f"Total inference time: {total_time:.2f}s")
        print(f"Total load time: {total_load_time:.2f}s")
        print(f"Average load time: {1000*total_load_time / total_images:.2f}ms")
        print(f"Average inference time: {avg_time * 1000:.2f}ms")
        print(f"FPS: {1 / avg_time:.2f}")

    # Save results to JSON
    results_output_path = os.path.join(args.output, "results.json")
    Path(results_output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(results_output_path, "w") as f:
        json.dump(all_results, f, indent=4)

    if args.eval:
        args.results = results_output_path
        args.res_type = "coco"
        coco_eval = evaluate(args)  # Pass args to evaluate function

        norm_fps = min(1 / avg_time, 25) / 25
        print(f"Normalized FPS: {norm_fps}")
        metric = (2 * norm_fps * coco_eval.stats[20]) / (norm_fps + coco_eval.stats[20])
        print(f"Final Metric: {metric}")


if __name__ == "__main__":
    main()


# python infer.py --model dfine --input /mlcv3/WorkingSpace/Personal/baotg/AICity25/Track4/data/collections/prepared/original/tests/ --output tests/dfine_original_640_60e/ --checkpoint ../training/dfine/trained/fisheye_80_original_640/best_stg1.pth --config ../training/dfine/configs/dfine/fisheye_80_original_640.yml --eval
