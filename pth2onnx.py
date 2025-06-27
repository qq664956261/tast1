import torch
import os
from models.SuperPoint import SuperPointNet # 你保存的模型定义模块

def export_superpoint_to_onnx(pth_path: str, onnx_path: str, input_size=(1, 3, 400, 800)):
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

    print(f"🟢 导出模型到 ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["score", "descriptor"],
        opset_version=11,
        do_constant_folding=True,

    )

    print("✅ ONNX 导出成功！")


if __name__ == "__main__":
    # 你自己的路径
    pth_model_path = "/home/zc/code/tast_1/weights/superpoint_v1.pth"
    output_onnx_path = "/home/zc/code/tast_1/weights/superpoint.onnx"
    export_superpoint_to_onnx(pth_model_path, output_onnx_path)
