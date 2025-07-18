# 🎯 Model Training Setup & Execution Guide

This guide outlines the full process for setting up the data structure, building a Docker environment, and executing training within a containerized environment.

---

## 📁 Directory Structure

Set up the following directory structure to organize your model checkpoints and training data:

```
data/
├── testset/                             # Images used for testing
├── ckpts/
│   └── dfine_l_obj2coco_e25.pth         # Pretrained checkpoint
└── training/
    ├── images/
    │   └── camera10_A_354.png           # Example training image
    ├── train.json                       # Training annotations
    ├── val.json                         # Validation annotations
    └── test.json                        # Test annotations
```

### 🔗 Downloads

- **📦 Pretrained Checkpoint**
  Download the required model checkpoint from:
  [dfine_l_obj2coco_e25.pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth)

- **📁 Training Data**
  Download the dataset (including `images/`, `train.json`, `val.json`, `test.json`) from:
  `[Insert Your Dataset URL Here]`

> ⚠️ Make sure to place the downloaded checkpoint under `data/ckpts/` and the dataset under `data/training/`.

---

## 🐳 Build the Docker Image

Use the following command to build the Docker image from the provided Dockerfile:

```bash
docker build -t track4 -f Dockerfile.train .
```

---

## 🚀 Run the Docker Container

Mount the `data/` directory into the container and enable GPU access:

```bash
docker run -it \
  -v /path/to/data:/data \
  --gpus all \
  --name test_track4 \
  track4:latest bash
```

> 💡 Replace `/path/to/data` with the absolute path to your `data/` directory.

---

## 🏋️‍♂️ Launch the Training

Once inside the container, start the training process by running:

```bash
# Pre-training
bash train.sh configs/dfine/L_pre_aip_vip_fsh_1600_pre.yml /data/ckpts/dfine_l_obj2coco_e25.pth

# Fine-tuning
bash train.sh configs/dfine/L_pre_aip_vip_fsh_1600_ft.yml trained/pretrained/last.pth
```

This command launches training with the specified configuration and checkpoint.

---

# 🚀 Inference Guide (Jetson)

This guide outlines the process for setting up the Docker environment on a Jetson device, converting the model, and running inference.

---

## 📁 Directory Structure

Ensure your `data` directory is structured as follows. The `.onnx` model file is required for conversion, and the `testset` contains the images for inference.

```
data/
├── testset/
│   └── image001.png                # Example test image
├── L_aip_vip_fsh_1600.onnx         # ONNX model for conversion
└── L_aip_vip_fsh_1600.engine       # Output TensorRT engine (generated later)
```

> ⚠️ Place your `.onnx` model and `testset/` images inside the `data/` directory before starting.

---

## 🐳 Build the Docker Image

Use the following command to build the Docker image specifically for the Jetson environment:

```bash
docker build -t track4_jetson -f Dockerfile.jetson .
```

---

## 🛰️ Run the Docker Container

Mount the `data/` directory and run the container with NVIDIA runtime access:

```bash
docker run -it \
  -v /path/to/data:/data \
  --ipc=host \
  --runtime=nvidia \
  --name test_track4_jetson \
  track4_jetson:latest bash
```

> 💡 Replace `/path/to/data` with the absolute path to your `data/` directory.

---

## ⚙️ Convert ONNX to TensorRT Engine

Once inside the container, convert the `.onnx` model to a highly optimized TensorRT engine. This command uses FP16 precision for better performance.

```bash
/usr/src/tensorrt/bin/trtexec --onnx=/data/L_aip_vip_fsh_1600.onnx --saveEngine=/data/L_aip_vip_fsh_1600.engine --fp16
```

---

## 🏁 Launch Inference

Finally, run the evaluation script using the newly generated TensorRT engine:

```bash
python run_evaluation_jetson.py --image_folder /data/testset/ --model_path /data/L_aip_vip_fsh_1600.engine
```

---

# Data Preparation

This section details the full pipeline for preparing the datasets required for both the pre-training and fine-tuning stages. The process involves downloading multiple source datasets, generating pseudo labels, and applying data augmentation.

-----

## 📥 Dataset Downloads

First, download the following datasets which form the basis for our training data:

  - **AIPCUP 2020:** [https://www.kaggle.com/datasets/awsaf49/aip-cup-2020-challenging-dataset/data](https://www.kaggle.com/datasets/awsaf49/aip-cup-2020-challenging-dataset/data)
  - **VIPCUP 2020:** [https://www.kaggle.com/datasets/awsaf49/vip-cup-2020](https://www.kaggle.com/datasets/awsaf49/vip-cup-2020)
  - **FishEye8K:** A large-scale fisheye camera dataset.
  - **FishEye1K:** A smaller fisheye camera dataset.

> 📝 Please organize the downloaded datasets into a logical directory structure for easier access in the following steps.

-----

## 🏷️ Step 1: Pseudo Labeling

We use a pre-trained model to generate initial object detection labels (pseudo labels) for the AIPCUP, VIPCUP, and FishEye datasets.

1.  **Download the VNPT Model & Code**
    Based on the source code released by [VNPT](https://github.com/vnptai/AICITY2024_Track4/), download the repository and its checkpoints. Place the entire contents into a folder named `VNPT` located at `prepare/pseudo/`.

2.  **Generate Pseudo Labels**
    Navigate to the `prepare/pseudo` directory. Before executing, you **must edit `pseudo.sh`** to set the correct paths for your downloaded datasets and define the `output_folder` where the results will be stored (e.g., `aip_vip_fisheye`).

    ```bash
    cd prepare/pseudo
    bash pseudo.sh
    ```

    This script will process the images and generate a `pseudo.json` file containing the annotations.

3.  **Convert Labels to YOLO Format**
    For easier processing, convert the generated `pseudo.json` annotations into the YOLO `.txt` format (one file per image).

    ```bash
    python prediction_to_yolo.py --image_folder <path/to/aip_vip_fisheye_images> \
                                 --predictions <path/to/pseudo.json> \
                                 --output_folder <path/to/aip_vip_fisheye_labels>
    ```

    > 💡 Replace the placeholders with the actual paths to your combined images, the `pseudo.json` file, and the target directory for the `.txt` labels.

-----

## 🛠️ Step 2: Generating the Fine-tuning Dataset

This script splits the pseudo-labeled data into `train.json` and `val.json` files, which are required for the fine-tuning process.

  - **Split Data for Fine-tuning:**
    Run the following command, pointing it to the images and the YOLO-formatted labels you just created.

    ```bash
    python prepare_pretrain_data_all.py --images <path/to/aip_vip_fisheye_images> \
                                        --labels <path/to/aip_vip_fisheye_labels> \
                                        --output <path/to/directory/save/split/annotations> \
                                        --type dfine
    ```

    The output directory will contain the `train.json`, `val.json`, and `test.json` files needed for the `data/training/` directory as described in the main training guide.

-----

## 🧬 Step 3: Generating the Pre-training Dataset

The pre-training dataset is enriched with copy-paste augmentation applied to the FishEye datasets to improve model generalization. The working directory for these commands is `prepare/`.

1.  **Augment FishEye Datasets with Copy-Paste**
    This script creates new training samples by copying objects from various images and pasting them onto others.

    ```bash
    # Augment FishEye8K
    python copy_paste.py --images <path/to/fe8k_images> \
                         --labels <path/to/fe8k_labels> \
                         --output <augmented_fe8k_output_path> \
                         --max_generations 30

    # Augment FishEye1K
    python copy_paste.py --images <path/to/fe1k_images> \
                         --labels <path/to/fe1k_labels> \
                         --output <augmented_fe1k_output_path> \
                         --max_generations 30
    ```

2.  **Create Final Pre-training Data**
    Combine the original pseudo-labeled data with the newly augmented FishEye data to create the final, comprehensive dataset for pre-training.

    ```bash
    python prepare_pretrain_data_all_replace.py \
      --images <path/to/aip_vip_fisheye_images> \
      --labels <path/to/aip_vip_fisheye_labels> \
      --output <path/to/final_pretrain_data_output> \
      --additional_images <path/to/augmented_fe1k_images> <path/to/augmented_fe8k_images> \
      --additional_labels <path/to/augmented_fe1k_labels> <path/to/augmented_fe8k_labels>
    ```

    This script will generate the final `train.json`, `val.json`, and `test.json` files for the pre-training phase.