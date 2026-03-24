from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# 让 Python 能找到当前目录下的模块
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

# 按你现在的代码先用 CoNet
from archs.co_lut_arch import CoNet


def extract_state_dict(ckpt):
    """
    尽量兼容不同权重保存格式
    """
    if not isinstance(ckpt, dict):
        return ckpt

    for key in ["params_ema", "params", "state_dict", "model", "net_g"]:
        if key in ckpt:
            return ckpt[key]

    # 有些权重本身就是参数字典
    return ckpt


def extract_output_tensor(model_output):
    """
    尽量兼容不同模型输出格式
    """
    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, (list, tuple)):
        for item in model_output:
            if isinstance(item, torch.Tensor):
                return item

    if isinstance(model_output, dict):
        for key in ["output", "out", "result", "img", "pred"]:
            if key in model_output and isinstance(model_output[key], torch.Tensor):
                return model_output[key]
        for _, value in model_output.items():
            if isinstance(value, torch.Tensor):
                return value

    raise TypeError(f"Unsupported model output type: {type(model_output)}")


def save_image_tensor(output_tensor: torch.Tensor, output_path: Path) -> None:
    """
    把 [1,C,H,W] 或 [C,H,W] tensor 保存成图片
    """
    if output_tensor.dim() == 4:
        output_tensor = output_tensor[0]

    if output_tensor.dim() != 3:
        raise ValueError(f"Unexpected output tensor shape: {tuple(output_tensor.shape)}")

    output_tensor = output_tensor.detach().float().cpu()

    # 限制到 0~1
    output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

    # CHW -> HWC
    out_img = output_tensor.permute(1, 2, 0).numpy()

    # 0~1 -> 0~255
    out_img = (out_img * 255.0).round().clip(0, 255).astype(np.uint8)

    # 若是 RGB，转 BGR 保存
    if out_img.ndim == 3 and out_img.shape[2] == 3:
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(output_path), out_img)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed, output path: {output_path}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Output file not created correctly: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, required=True, help="输出图片路径")
    parser.add_argument("--weights", type=str, required=True, help="权重文件路径")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    weights_path = Path(args.weights).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] input = {input_path}")
    print(f"[INFO] output = {output_path}")
    print(f"[INFO] weights = {weights_path}")

    # 1. 构建模型
    model = CoNet()

    # 2. 加载权重
    ckpt = torch.load(str(weights_path), map_location=device)
    state_dict = extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"[INFO] missing keys: {len(missing)}")
    print(f"[INFO] unexpected keys: {len(unexpected)}")
    if missing:
        print(f"[DEBUG] missing sample: {missing[:10]}")
    if unexpected:
        print(f"[DEBUG] unexpected sample: {unexpected[:10]}")

    model.to(device)
    model.eval()

    # 3. 读图
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read input image: {input_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # 4. 推理
    with torch.no_grad():
        raw_output = model(img_tensor)

    output_tensor = extract_output_tensor(raw_output)
    print(f"[INFO] output tensor shape: {tuple(output_tensor.shape)}")

    # 5. 保存
    save_image_tensor(output_tensor, output_path)

    print(f"SUCCESS: {output_path}")


if __name__ == "__main__":
    main()