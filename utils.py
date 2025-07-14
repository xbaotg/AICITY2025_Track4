import json
from collections import OrderedDict

import cv2
import numpy as np
import tensorrt as trt
import torch
from models.base import BaseModel


class DFineTRTModel(BaseModel):
    """
    Optimized Class for the DFine model using TensorRT, corrected for dynamic shapes.
    """

    _NP_TO_TORCH_DTYPE_MAP = {
        np.float32: torch.float32,
        np.float16: torch.float16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.int8: torch.int8,
        np.bool_: torch.bool,
    }

    def __init__(self, max_batch_size=32, verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_batch_size = max_batch_size
        self.logger = (
            trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        )
        trt.init_libnvinfer_plugins(self.logger, "")

        self.engine = None
        self.context = None
        self.bindings_gpu_ptrs = []
        self.gpu_buffers = OrderedDict()
        self.input_names = []
        self.output_names = []
        self.input_size = None
        self.classes = None
        self.orig_size_gpu = torch.empty((1, 2), dtype=torch.int64, device="cuda:0")

    def load_engine(self, path):
        print(f"Loading engine from {path}")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _get_io_names(self, io_mode):
        names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == io_mode:
                names.append(name)
        return names

    def get_input_names(self):
        return self._get_io_names(trt.TensorIOMode.INPUT)

    def get_output_names(self):
        return self._get_io_names(trt.TensorIOMode.OUTPUT)

    def allocate_buffers(self):
        """
        Allocates GPU memory for I/O buffers based on the engine's max profile shape.
        This should be called once after loading the engine.
        """
        print("Allocating GPU buffers based on max profile shapes...")
        self.bindings_gpu_ptrs = [None] * self.engine.num_io_tensors

        # Use the active optimization profile (usually 0).
        # profile_index = (
        #     self.engine.active_optimization_profile
        #     if self.engine.active_optimization_profile >= 0
        #     else 0
        # )
        profile_index = 0  # Assuming single profile for simplicity

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            profile_shapes = self.engine.get_tensor_profile_shape(name, profile_index)
            # Allocate buffer using the maximum shape from the profile.
            max_shape = profile_shapes[2]  # Index 2 is max_shape

            print(f"  - Allocating buffer for '{name}' with max shape: {max_shape}")

            dtype = self._NP_TO_TORCH_DTYPE_MAP[
                trt.nptype(self.engine.get_tensor_dtype(name))
            ]
            buffer_tensor = torch.empty(
                tuple(max_shape), dtype=dtype, device="cuda:0"
            ).contiguous()

            self.gpu_buffers[name] = buffer_tensor
            self.bindings_gpu_ptrs[i] = buffer_tensor.data_ptr()

    def load_model(
        self, checkpoint_path: str, input_size: int = 1600, type_class: int = 5
    ):
        self.input_size = input_size
        self.engine = self.load_engine(checkpoint_path)
        if not self.engine:
            raise RuntimeError("Failed to load TensorRT engine.")

        self.context = self.engine.create_execution_context()
        if not self.context:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()

        # Allocate buffers robustly based on the engine's max profile shape.
        self.allocate_buffers()

        if type_class == 5:
            self.classes = {0: "bus", 1: "bike", 2: "car", 3: "pedestrian", 4: "truck"}
        else:
            self.classes = {5: "bus", 3: "bike", 2: "car", 0: "pedestrian", 7: "truck"}

    def preprocess_image(self, img_np: np.ndarray):
        img_resized = cv2.resize(
            img_np, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR
        )
        im_tensor = (
            torch.as_tensor(img_resized, device="cuda:0")
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div_(255.0)
        )
        return im_tensor

    def inference(self, img_np: np.ndarray, conf_thres: float = 0.25):
        # 1. Preprocess the image
        # Set original image size (H, W). np.shape is (H, W, C).
        self.orig_size_gpu[0, 0] = img_np.shape[0]
        self.orig_size_gpu[0, 1] = img_np.shape[1]
        im_data_processed = self.preprocess_image(img_np)

        # 2. Prepare input dictionary
        input_feed = {
            self.input_names[0]: im_data_processed,
            self.input_names[1]: self.orig_size_gpu,
        }

        # 3. Set dynamic input shapes and copy data to GPU buffers
        for name, tensor_data in input_feed.items():
            self.context.set_input_shape(name, tensor_data.shape)
            self.gpu_buffers[name].copy_(tensor_data)

        # 4. Run inference
        self.context.execute_v2(bindings=self.bindings_gpu_ptrs)

        # 5. Get outputs by slicing the buffers to the actual output shape
        output_tensors = {}
        for name in self.output_names:
            output_shape = self.context.get_tensor_shape(name)
            output_tensors[name] = self.gpu_buffers[name][
                tuple(slice(0, dim) for dim in output_shape)
            ]

        if not output_tensors:
            return []

        # 6. Post-process the valid results
        boxes = output_tensors["boxes"][0].cpu()
        labels = output_tensors["labels"][0].cpu()
        scores = output_tensors["scores"][0].cpu()

        valid_indices = scores >= conf_thres
        filtered_boxes = boxes[valid_indices].tolist()
        filtered_labels = labels[valid_indices].tolist()
        filtered_scores = scores[valid_indices].tolist()

        results = [
            {"bbox": b, "cls": c, "conf": s}
            for b, c, s in zip(filtered_boxes, filtered_labels, filtered_scores)
        ]

        # print(results)

        return results

    def __del__(self):
        # Clean up CUDA memory and TensorRT objects
        for buffer in self.gpu_buffers.values():
            del buffer
        if hasattr(self, "context"):
            del self.context
        if hasattr(self, "engine"):
            del self.engine


def f1_score(predictions_path, ground_truths_path):
    coco_gt = COCO(ground_truths_path)

    gt_image_ids = coco_gt.getImgIds()

    with open(predictions_path, "r") as f:
        detection_data = json.load(f)
    filtered_detection_data = [
        item for item in detection_data if item["image_id"] in gt_image_ids
    ]
    with open("./temp.json", "w") as f:
        json.dump(filtered_detection_data, f)
    coco_dt = coco_gt.loadRes("./temp.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Assuming the F1 score is at index 20 in the stats array
    return coco_eval.stats[20]  # Return the F1 score from the evaluation stats
    # return 0.85  # Simulated constant value for demo purposes


def imread(filename: str, flags: int = cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(filename, np.uint8), flags)


def get_model(model_path):
    model = DFineTRTModel()
    model.load_model(model_path)
    return model


def preprocess_image(image_path):
    pass


def postprocess_result(results):
    pass


def changeId(id):
    sceneList = ["M", "A", "E", "N"]
    cameraId = int(id.split("_")[0].split("camera")[1])
    sceneId = sceneList.index(id.split("_")[1])
    frameId = int(id.split("_")[2])
    imageId = int(str(cameraId) + str(sceneId) + str(frameId))
    return imageId