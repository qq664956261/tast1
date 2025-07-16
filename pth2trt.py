import torch
import os
from models.SuperPoint import SuperPointNet # 你保存的模型定义模块
import tensorrt as trt
import openvino as ov
def export_superpoint_to_trt(pth_path: str, trt_path: str, input_size=(1, 1, 400, 800)):
    """
    将 SuperPoint 模型从 PyTorch .pth 导出为 ONNX 格式

    :param pth_path: .pth 权重路径
    :param onnx_path: 导出的 .onnx 保存路径
    :param input_size: 模型输入尺寸 (B, C, H, W)
    """
    assert os.path.exists(pth_path), f"模型文件不存在: {pth_path}"

    # 加载模型
    model = SuperPointNet()
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(*input_size)  # 通常是 1×1×240×320，灰度图
    output_onnx_path = "/home/zc/code/tast_1/weights/superpoint.onnx"
    print(f"🟢 导出模型到 ONNX: {output_onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        input_names=["input"],
        output_names=["score", "descriptor"],
        opset_version=11,
        do_constant_folding=True
    )

    print("✅ ONNX 导出成功！")


    ov_model = ov.convert_model("/home/zc/code/tast_1/weights/superpoint.onnx", example_input=dummy_input)
    ov.save_model(ov_model, "/home/zc/code/tast_1/weights/superpoint.xml")
    print("✅ openvino 导出成功！")



    # export to tensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(output_onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise Exception("Failed to parse the ONNX file")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    serialized_engine = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    trt_engine_path = trt_path
    with open(trt_engine_path, "wb") as f:
        f.write(engine.serialize())


if __name__ == "__main__":
    # 你自己的路径
    pth_model_path = "/home/zc/code/tast_1/weights/superpoint_v1.pth"
    output_trt_path = "/home/zc/code/tast_1/weights/superpoint.trt"
    export_superpoint_to_trt(pth_model_path, output_trt_path)
