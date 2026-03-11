import argparse
import os
import sys
import cv2
import numpy as np
import torch

# 【防暴毙机制】：将当前目录加入环境变量，防止 Python 找不到 archs 文件夹
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# =====================================================================
# ⚠️【请核对】：下面这行是唯一可能需要你微调的地方
# 请你双击打开 archs/co_lut_arch.py 这个文件，滑到最底下或最上面。
# 看看那个被定义为 nn.Module 的主类名叫什么？
# 它可能叫 CoLUT、CoTF、或者 CoTFArch。把下面这行的 CoLUT 换成真实的类名。
# =====================================================================
from archs.co_lut_arch import CoNet  # <--- 修改这里


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, required=True, help="输出图片路径")
    parser.add_argument("--weights", type=str, required=True, help="权重文件路径")
    args = parser.parse_args()

    # 1. 准备显卡 (强依赖CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 召唤大厨并穿上装备（实例化网络并加载权重）
    model = CoNet()  # <--- 如果上面改了名字，这里也要同步改。
    
    # 学术界的权重经常包在一层字典里，这里做一层兼容剥离
    state_dict = torch.load(args.weights, map_location=device)
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']
        
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval() # 切换到推理模式

    # 3. 读取图片并预处理（OpenCV 默认 BGR，转为 RGB 并归一化到 0~1）
    img = cv2.imread(args.input)
    if img is None:
        raise ValueError(f"无法读取输入图片，请检查路径: {args.input}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    # 4. 运行算法（不计算梯度，省显存）
    with torch.no_grad():
        output_tensor = model(img_tensor)

    # 5. 后处理并出锅（Tensor 转回 numpy 数组，乘以 255，并限制在 0-255 范围内）
    out_img = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)
    
    # 存回硬盘
    cv2.imwrite(args.output, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print("SUCCESS")

if __name__ == "__main__":
    main()