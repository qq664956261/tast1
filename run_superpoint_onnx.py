#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_superpoint_onnx.py

演示：用 ONNX Runtime 加载 SuperPoint ONNX 模型，
并对一张灰度图（或随机 tensor）做推理，输出 score/descriptor 的信息。
"""

import onnxruntime as rt
import numpy as np
from PIL import Image
import argparse


def preprocess(img_path: str, target_size=(400, 800)) -> np.ndarray:
    """
    1. 用 Pillow 读入灰度图
    2. resize 到 (H, W) = target_size
    3. 归一化到 [0,1]，变成 float32
    4. 增加 batch/chan 维度，返回 shape=(1,1,H,W)
    """
    img = Image.open(img_path).convert('L')
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # 增加 batch 和 channel 维度
    return arr[np.newaxis, np.newaxis, :, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True,
                        help="ONNX 模型路径，比如 ./superpoint.onnx")
    parser.add_argument('--img', type=str, default=None,
                        help="（可选）测试用灰度图路径，不指定则用随机 tensor")
    args = parser.parse_args()

    # 1. 创建 ONNX Runtime 会话
    sess = rt.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])

    # 2. 打印 input/output 的名字 & 顺序
    input_name   = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    print(f"输入 tensor 名称: {input_name}")
    print(f"输出 tensor 列表: {output_names}\n")

    # 3. 构造输入
    if args.img:
        inp = preprocess(args.img)          # shape=(1,1,400,800)
    else:
        inp = np.random.randn(1, 1, 400, 800).astype(np.float32)

    # 4. 推理
    outputs = sess.run(output_names, {input_name: inp})

    # ONNX 定义里通常是 [score, descriptor]
    score, desc = outputs[0], outputs[1]

    # 5. 打印结果
    print(f"score  shape = {score.shape}, mean = {score.mean():.6f}")
    print(f"desc   shape = {desc.shape}, mean = {desc.mean():.6f}")
    d = desc[0].reshape(256, -1)       # 拉成 (256, 5000)
    norms = np.linalg.norm(d, axis=0)  # 每列是一条 descriptor 的范数
    print("L2-norm   mean =", norms.mean())
    print("L2-norm std  =", norms.std())
    #print(desc)


if __name__ == '__main__':
    main()