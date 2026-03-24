import argparse
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as F

# 防暴毙机制
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.EVSSM import EVSSM


def prepare_single_image_dataset(input_image_path: str, temp_dir: str):
    """
    为 EVSSM 创建单张图片的 input/target 目录结构。
    
    Args:
        input_image_path: 输入图像路径
        temp_dir: 临时数据集目录
    """
    temp_dir = Path(temp_dir)
    test_dir = temp_dir / "test"
    input_dir = test_dir / "input"
    target_dir = test_dir / "target"
    
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制输入图到 input 目录
    input_image_path = Path(input_image_path).resolve()
    filename = input_image_path.name
    
    # 使用 OpenCV 读取并存储，避免编码问题
    img = cv2.imread(str(input_image_path))
    if img is None:
        raise ValueError(f"无法读取输入图片: {input_image_path}")
    
    cv2.imwrite(str(input_dir / filename), img)
    
    # target 目录放一个占位图（EVSSM 需要但实际不会使用）
    cv2.imwrite(str(target_dir / filename), img)
    
    return filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, required=True, help="输出图片路径")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    args = parser.parse_args()
    
    # 1. 验证输入文件
    if not os.path.exists(args.input):
        raise ValueError(f"无法读取输入图片: {args.input}")
    
    if not os.path.exists(args.model):
        raise ValueError(f"无法读取模型权重: {args.model}")
    
    # 2. 创建临时数据集目录
    temp_dir = Path(args.output).parent / "evssm_temp_dataset"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. 【关键】准备单张图片的 input/target 结构
    filename = prepare_single_image_dataset(args.input, str(temp_dir))
    
    # 4. 准备输出目录
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EVSSM()
    model = model.to(device)
    
    # 6. 加载权重（嵌套在 'params' 字典中）
    state_dict = torch.load(args.model, map_location=device)
    if 'params' in state_dict:
        state_dict = state_dict['params']
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # 7. 读取输入图片
    input_image = Image.open(args.input).convert('RGB')
    input_tensor = F.to_tensor(input_image).unsqueeze(0).to(device)
    
    # 8. 运行推理
    with torch.no_grad():
        pred = model(input_tensor)
    
    # 9. 后处理并保存
    pred_clip = torch.clamp(pred, 0, 1)
    pred_clip += 0.5 / 255
    
    output_image = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
    output_image.save(args.output)
    
    print("SUCCESS")


if __name__ == "__main__":
    main()
