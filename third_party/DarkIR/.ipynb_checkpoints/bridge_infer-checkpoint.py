import os
import sys
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# 强制将当前目录加入系统路径，防止找不到 archs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 【修正 1】导入真正的 DarkIR 模型
from archs.DarkIR import DarkIR

def parse_args():
    parser = argparse.ArgumentParser(description="DarkIR Inference Bridge")
    parser.add_argument("--input", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, required=True, help="输出图片路径")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    # 【修正 2】使用 action="store_true"。前端传了这个参数就是开启缩放，不传就是不缩放
    parser.add_argument("--resize", action="store_true", help="是否对超大图进行降采样处理")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 验证输入文件
    if not os.path.exists(args.input):
        raise ValueError(f"无法读取输入图片: {args.input}")
    if not os.path.exists(args.model):
        raise ValueError(f"无法读取模型权重: {args.model}")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 初始化 DarkIR 并加载权重
    model = DarkIR()
    checkpoint = torch.load(args.model, map_location=device)
    
    # 兼容 DarkIR 官方权重的字典格式
    if 'params' in checkpoint:
        model.load_state_dict(checkpoint['params'], strict=True)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
        
    model.eval()
    model.to(device)
    
    # 3. 读取输入图片 (使用 OpenCV 统一通道和格式，比 PIL 更稳)
    img = cv2.imread(args.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    
    # 记录最原始的尺寸，留作最后还原用
    _, _, orig_H, orig_W = tensor.shape
    
    # 4. 【大图处理策略：缩放】
    # 如果开启了 resize 参数，且宽高大于等于 1500，则长宽各缩小一半
    if args.resize and (orig_H >= 1500 or orig_W >= 1500):
        new_size = (orig_H // 2, orig_W // 2)
        tensor = F.interpolate(tensor, size=new_size, mode='bilinear', align_corners=False)
        
    # 记录准备进入网络的特征图尺寸（用于后面切掉 Padding）
    _, _, current_H, current_W = tensor.shape
    
    # 5. 动态 Padding (DarkIR 这种 Metaformer 架构通常需要尺寸是 32 的倍数)
    pad_factor = 32
    pad_h = (pad_factor - current_H % pad_factor) % pad_factor
    pad_w = (pad_factor - current_W % pad_factor) % pad_factor
    if pad_h != 0 or pad_w != 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), 'reflect')
        
    # 6. 模型推理
    with torch.no_grad():
        output = model(tensor)
        # 防止有的模型输出元组
        if isinstance(output, (list, tuple)):
            output = output[0]
            
    # 7. 【修正 3】必须先裁掉 Padding，再进行放大！
    if pad_h != 0 or pad_w != 0:
        output = output[:, :, :current_H, :current_W]
        
    # 8. 如果之前缩小过，现在放大回最原始的分辨率
    if args.resize and (orig_H >= 1500 or orig_W >= 1500):
        output = F.interpolate(output, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
        
    # 9. 后处理并保存
    output = torch.clamp(output, 0, 1)
    output_img = output.squeeze(0).cpu().numpy()
    output_img = np.transpose(output_img, (1, 2, 0)) # CHW -> HWC
    output_img = (output_img * 255.0).round().astype(np.uint8)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR) # 存图前转回 BGR
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, output_img)
    
    print("SUCCESS")

if __name__ == "__main__":
    main()