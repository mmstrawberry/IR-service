import os
import sys
import cv2
import argparse
import torch
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

# 防暴毙机制
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from archs.retinexformer import RetinexFormer


def pad_tensor(tensor, multiple=8):
    """填充张量使其尺寸为 multiple 的倍数"""
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, required=True, help="输出图片路径")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--resize", type=bool, default=False, help="是否对大图片进行缩小处理")
    args = parser.parse_args()
    
    # 1. 验证输入文件
    if not os.path.exists(args.input):
        raise ValueError(f"无法读取输入图片: {args.input}")
    
    if not os.path.exists(args.model):
        raise ValueError(f"无法读取模型权重: {args.model}")
    
    # 2. 准备设备和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetinexFormer()
    model = model.to(device)
    model.eval()
    
    # 3. 加载权重（嵌套在 'params' 字典中）
    checkpoints = torch.load(args.model, map_location=device, weights_only=False)
    weights = checkpoints['params']
    # 添加 'module.' 前缀以适配 DataParallel 权重格式
    weights = {'module.' + key: value for key, value in weights.items()}
    model.load_state_dict(weights)
    
    # 4. 读取输入图片
    pil_to_tensor = transforms.ToTensor()
    img = Image.open(args.input).convert('RGB')
    tensor = pil_to_tensor(img).unsqueeze(0).to(device)
    
    # 5. 获取原始尺寸
    _, _, H, W = tensor.shape
    
    # 6. 【可选】处理超大图片 (>= 1500)
    if args.resize and (H >= 1500 or W >= 1500):
        new_size = [int(dim // 2) for dim in (H, W)]
        downsample = torch.nn.functional.interpolate
        tensor = torch.nn.functional.interpolate(tensor, size=new_size, mode='bilinear', align_corners=False)
    
    # 7. 填充张量以适应网络
    tensor = pad_tensor(tensor)
    
    # 8. 运行推理
    with torch.no_grad():
        output = model(tensor, side_loss=False)
    
    # 9. 上采样回原始分辨率
    if args.resize and (H >= 1500 or W >= 1500):
        output = torch.nn.functional.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
    
    # 10. 后处理
    output = torch.clamp(output, 0., 1.)
    output = output[:, :, :H, :W]
    
    # 11. 保存输出图片
    tensor_to_pil = transforms.ToPILImage()
    output_img = tensor_to_pil(output.squeeze(0).cpu())
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_img.save(args.output)
    
    print("SUCCESS")


if __name__ == "__main__":
    main()
