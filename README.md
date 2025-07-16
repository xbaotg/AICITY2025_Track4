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
