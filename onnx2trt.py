import os
import tensorrt as trt
import onnx
#nvidia_tensorrt-8.4.3.1
def convert_onnx_to_trt(onnx_path: str,
                        trt_path: str,
                        max_workspace_size: int = 1 << 30,
                        fp16: bool = False,
                        input_shape=(1, 3, 400, 800)):
    """
    将 ONNX 模型转为 TensorRT 引擎并保存到文件，带动态输入优化配置
    :param onnx_path: ONNX 模型路径
    :param trt_path: 保存 TRT 引擎路径
    :param max_workspace_size: 最大临时显存（字节）
    :param fp16: 是否启用 FP16 加速
    :param input_shape: ONNX 中 'input' 张量的 (min/opt/max) shape 示例
    """
    assert os.path.exists(onnx_path), f"ONNX 文件不存在: {onnx_path}"
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # 1. 创建 Builder/Network/Parser
    builder = trt.Builder(TRT_LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 2. 解析 ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("❌ ONNX 解析失败")

    # 3. 创建 BuilderConfig 并设置工作区
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 4. **创建并添加 Optimization Profile**  
    profile = builder.create_optimization_profile()
    # 假设你的网络第一个输入名叫 "input"
    # 你需要根据实际导出的 ONNX model.input_names 来填写
    min_shape = input_shape    # e.g. (1,1,240,320)
    opt_shape = input_shape    # e.g. (1,1,240,320) —— 日常推理时常用尺寸
    max_shape = (1, 3, 400, 800)  # 可支持的最大尺寸，比如 2x 240×320
    profile.set_shape("input", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    # 5. 构建 TensorRT 引擎
    print(f"🟢 正在构建 TRT 引擎 (FP16={fp16}) …")
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("❌ TensorRT 引擎构建失败")

    # 6. 序列化并保存
    with open(trt_path, 'wb') as f:
        f.write(engine.serialize())
    print(f"✅ 已保存 TRT 引擎到: {trt_path}")


if __name__ == "__main__":
    onnx_model = "/home/zc/code/tast_1/weights/superpoint.onnx"
    trt_engine  = "/home/zc/code/tast_1/weights/superpoint.trt"
    m = onnx.load(onnx_model)
    try: 
        onnx.checker.check_model(m) 
    except Exception: 
        print("Model incorrect") 
    else: 
        print("Model correct")
    print([ (o.name, o.type.tensor_type.shape) for o in m.graph.output ])
    # 注意：input_shape、max_shape 要与你的模型导出时的 dynamic_axes 保持一致
    convert_onnx_to_trt(
        onnx_model,
        trt_engine,
        fp16=False,
        input_shape=(1, 3, 400, 800)  # min/opt shape
    )
