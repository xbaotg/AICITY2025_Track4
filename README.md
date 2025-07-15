# 🎯 Model Training Setup & Execution Guide

This guide outlines the full process for setting up the data structure, building a Docker environment, and executing training within a containerized environment.

---

## 📁 Directory Structure

Set up the following directory structure to organize your model checkpoints and training data:

```
data/
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

* **📦 Pretrained Checkpoint**
  Download the required model checkpoint from:
  [dfine\_l\_obj2coco\_e25.pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_obj2coco_e25.pth)

* **📁 Training Data**
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

docker run -it -v /mlcv3/WorkingSpace/Personal/baotg/AICity25/Track4/release/data:/data --gpus all --name test_track4_jetson track4_jetson:latest bash

/usr/src/tensorrt/bin/trtexec --onnx=/data/L_aip_vip_fsh_1600.onnx --saveEngine=/data//L_aip_vip_fsh_1600.engine --fp16


<!-- # 🐳 Docker Setup for AI City ICCV 2025 Track 4

This guide provides a quick start tutorial for container submissions using a fine-tuned YOLOv11n model as a reference.

**You should develop your own implementations of the get_model, preprocess_image, and postprocess_results functions in utils.py for your submission. Prize winners need to place their pretraining data (if applicable) and models in the shared Google Drive and upload training and evaluation containers on Docker Hub. Your training and evaluation scripts inside the container should load models from the mounted /models directory and the data from the /data directory.**

# Evaluation Container Instruction

## 🔹 Pull the Prebuilt Docker Image

Start by pulling the prebuilt Docker image designed for Jetson devices:

```bash
docker pull ganzobtn/aicity_iccv_2025_track4_jetson:latest
```
## 🔹 Build Image Locally (Optional)
If you'd prefer to build the Docker image locally:

```bash
docker build -f Dockerfile.jetson -t ganzobtn/aicity_iccv_2025_track4_jetson:latest .
```

## 🔹 Run the Docker container

```bash
IMAGE="ganzobtn/aicity_iccv_2025_track4_jetson:latest"
DATA_DIR="/path/to/your/data"
```

```bash
docker run -it --ipc=host --runtime=nvidia -v ${DATA_DIR}:/data ${IMAGE}
```

📁 Expected Directory Structure

The `run_evaluation_jetson.py` script inside the container expects the following structure:

- `/data/FishEye1K_eval/` Contains the groundtruth.json file, evaluation images and corresponding annotation files.

- `models/yolo11n_fisheye8k.pt`  
  The fine-tuned YOLOv11n model file used for inference.
  


# Training Container Instruction

This section provides a getting started guide for setting up and running the training Docker container for the challenge, which uses a YOLOv11n model finetuning pipeline.

## 🔹 Pull Prebuilt Docker Image 

You can use the prebuilt Docker image available on Docker Hub:

```bash
docker pull ganzobtn/aicity_iccv_2025_track4_train:latest
```
---

## 🔹 Build the Docker Image Locally (Optional)
If you'd prefer to build the image from the provided Dockerfile:

```bash
docker build -f Dockerfile.train -t ganzobtn/aicity_iccv_2025_track4_train:latest .
```

## 🔹 Run the Docker Container
Set your local paths and run the container:

```bash
IMAGE="ganzobtn/aicity_iccv_2025_track4_train:latest"

DATA_DIR="/path/to/your/data"
MODEL_DIR="/path/to/your/models"

docker run -it --ipc=host --runtime=nvidia \
  -v ${DATA_DIR}:/data \
  -v ${MODEL_DIR}:/models \
  ${IMAGE}
```

📁 Expected Directory Structure
 run_train_yolo.py script inside the container expects the following structure:

 - A dataset folder named Fisheye8K located inside /data.

 - Trained models and output logs will be saved to /models.
 -->
