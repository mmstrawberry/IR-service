import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

# 防暴毙机制：确保能找到 network 和 config
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 【修正 1】引入官方配置类和正确的类名
from config.basic import ConfigBasic
from network_level2 import Slot_model as Slot_model_level2
from network_level3 import Slot_model as Slot_model_level3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, required=True, help="输出图片路径")
    parser.add_argument("--weights", type=str, required=True, help="权重文件路径")
    parser.add_argument("--dataset", type=str, default="SICE", help="SICE, MSEC, LCDP")
    parser.add_argument("--level", type=int, default=2, help="2 或 3")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 【修正 2】初始化官方 Config 并手动设置必要参数
    cfg = ConfigBasic()
    cfg.level = args.level
    cfg.dataset = args.dataset
    cfg.device = device
    # 调用官方的 set_dataset 补充其他超参数
    cfg.set_dataset()

    # 1. 召唤网络并传入 cfg
    if args.level == 2:
        model = Slot_model_level2(cfg)
    else:
        model = Slot_model_level3(cfg)

    # 2. 加载权重 (官方格式在 'model' 键值对里)
    ckpt = torch.load(args.weights, map_location='cpu')
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    
    model.to(device)
    model.eval()

    # 3. 读取图片
    img = cv2.imread(args.input)
    if img is None: raise ValueError(f"无法读取图片: {args.input}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    
    # 【修正 3】官方同款 Padding 机制 (确保尺寸是 4 的倍数)
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    factor = 4
    padh = (factor - h % factor) % factor
    padw = (factor - w % factor) % factor
    if padh != 0 or padw != 0:
        img_tensor = F.pad(img_tensor, (0, padw, 0, padh), 'reflect')

    # 4. 运行推理 (按照官方 validate 函数的调用方式)
    with torch.no_grad():
        # 官方写法：model(x, x, inference=True)
        preds, _, _ = model(img_tensor, img_tensor, inference=True)
        
        # 裁剪掉刚才 Padding 的部分
        preds = preds[:, :, :h, :w]
        preds = torch.clamp(preds, 0, 1)

    # 5. 保存结果
    out_img = preds.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0
    out_img = out_img.astype(np.uint8)
    cv2.imwrite(args.output, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    
    print("SUCCESS")

if __name__ == "__main__":
    main()