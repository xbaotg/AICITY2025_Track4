import os
import time
import argparse
import json
from tqdm import tqdm
from .utils import get_model
import json
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/data/Fisheye1K_eval/images",
        help="Path to image folder",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/yolo11n_fisheye8k.pt",
        help="Path to the model",
    )
    parser.add_argument(
        "--max_fps", type=float, default=25.0, help="Maximum FPS for evaluation"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/data/Fisheye1K_eval/predictions.json",
        help="Output JSON file for predictions",
    )
    # parser.add_argument('--ground_truths_path', type=str, default='/data/Fisheye1K_eval/groundtruth.json', help='Path to ground truths JSON file')
    args = parser.parse_args()

    image_folder = args.image_folder
    model_path = args.model_path
    model = get_model(model_path)

    image_files = sorted(
        [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    print(f"Found {len(image_files)} images.")

    predictions = []
    print("Prediction started")
    total_time = 0
    start_time = time.time()
    for image_path in tqdm(image_files):
        img = cv2.imread(image_path)
        if img is None:
            continue

        t0 = time.time()
        # img = preprocess_image(img)
        results = model.inference(img)
        # results = postprocess_result(results)  # [boxes, scores, classes]
        predictions.append((image_path, results))
        t3 = time.time()
        total_time += t3 - t0

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Processed {len(image_files)} images in {elapsed_time:.2f} seconds.")
    print(f"Avg Processing Time           : {total_time/len(image_files)*1000:.2f} ms")

    predictions_json = []
    with open(args.output_json, "w") as f:
        json.dump(predictions_json, f, indent=2)

    fps = len(image_files) / total_time
    normfps = min(fps, args.max_fps) / args.max_fps

    print("\n--- Evaluation Complete ---")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Normalized FPS: {normfps:.4f}")


if __name__ == "__main__":
    main()
