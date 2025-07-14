import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def convert_onnx_to_trt(onnx_path, engine_path, batch_size=1, precision="fp16"):
    # Initialize TensorRT stuff
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    # Parse ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as model:
        parser.parse(model.read())

    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    # Create optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        min=(1, 3, 640, 640),
        opt=(batch_size, 3, 640, 640),
        max=(batch_size, 3, 640, 640),
    )
    profile.set_shape(
        "orig_target_sizes", min=(1, 2), opt=(batch_size, 2), max=(batch_size, 2)
    )
    # opset 17
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 2GB
    config.add_optimization_profile(profile)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        for layer_idx in range(network.num_layers):
            layer = network[layer_idx]
            if layer.type == trt.LayerType.NORMALIZATION:
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)

    config.set_flag(trt.BuilderFlag.FP16)
    # Build and save engine
    engine = builder.build_serialized_network(network, config)
    # engine = trt.runtime.deserialize_cuda_engine(plan)

    with open(engine_path, "wb") as f:
        print(engine)
        f.write(engine)

    f.close()
    print(f"Successfully converted model to TensorRT engine: {engine_path}")


if __name__ == "__main__":
    # Example usage
    convert_onnx_to_trt(
        onnx_path="/mlcv3/WorkingSpace/Personal/baotg/AICity25/Track4/source/training/dfine/trained/fisheye_80_original_640/best_stg2.onnx",
        engine_path="/mlcv3/WorkingSpace/Personal/baotg/AICity25/Track4/source/training/dfine/trained/fisheye_80_original_640/best_stg2_fp16_test.engine",
        batch_size=1,
        precision="fp16",
    )
