import argparse
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path

# 防暴毙机制
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from network_level2 import Exposure_Slot as Exposure_Slot_Level2
from network_level3 import Exposure_Slot as Exposure_Slot_Level3


def dynamic_update_dataset_txt(dataset: str, input_image_path: str, output_dir: str):
    """
    根据 dataset 动态替换对应的 Dataset_txt 文件。
    
    关键逻辑：
    - SICE 对应 SICEV2_Train.txt / SICEV2_Test.txt
    - MSEC 对应 MSEC_Train.txt / MSEC_Test.txt  
    - LCDP 对应 LCDP_Train.txt / LCDP_Test.txt
    
    Args:
        dataset: 数据集名称 (SICE, MSEC, LCDP)
        input_image_path: 输入图像路径
        output_dir: 输出目录
    """
    dataset_txt_dir = Path(__file__).parent / "Dataset_txt"
    dataset_txt_dir.mkdir(parents=True, exist_ok=True)
    
    # 根据 dataset 选择要替换的 txt 文件
    if dataset.upper() == "SICE":
        txt_files = ["SICEV2_Train.txt", "SICEV2_Test.txt"]
    elif dataset.upper() == "MSEC":
        txt_files = ["MSEC_Train.txt", "MSEC_Test.txt"]
    elif dataset.upper() == "LCDP":
        txt_files = ["LCDP_Train.txt", "LCDP_Test.txt"]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # 生成单行 filelist（输入图 | 输出目录）
    input_path_abs = Path(input_image_path).resolve()
    output_dir_abs = Path(output_dir).resolve()
    filelist_line = f"{str(input_path_abs)}|{str(output_dir_abs)}\n"
    
    # 动态替换所有对应的 txt 文件
    for txt_file in txt_files:
        txt_path = dataset_txt_dir / txt_file
        with open(txt_path, 'w') as f:
            f.write(filelist_line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, required=True, help="输出图片路径")
    parser.add_argument("--weights", type=str, required=True, help="权重文件路径")
    parser.add_argument("--dataset", type=str, default="SICE", 
                       help="数据集名称: SICE, MSEC, LCDP")
    parser.add_argument("--level", type=int, default=2, help="网络层级: 2 或 3")
    args = parser.parse_args()
    
    # 1. 准备显卡
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. 验证输入文件存在
    if not os.path.exists(args.input):
        raise ValueError(f"无法读取输入图片: {args.input}")
    
    if not os.path.exists(args.weights):
        raise ValueError(f"无法读取权重文件: {args.weights}")
    
    # 3. 创建临时输出目录（用于 DataLoader）
    temp_output_dir = Path(args.output).parent / "exposure_slot_temp"
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. 【关键】动态更新 Dataset_txt 中的 txt 文件
    dynamic_update_dataset_txt(args.dataset, args.input, str(temp_output_dir))
    
    # 5. 召唤网络并加载权重
    if args.level == 2:
        model = Exposure_Slot_Level2()
    elif args.level == 3:
        model = Exposure_Slot_Level3()
    else:
        raise ValueError(f"Unsupported level: {args.level}")
    
    # 加载权重（可能包在字典里）
    state_dict = torch.load(args.weights, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # 6. 读取和预处理图片 (OpenCV BGR -> RGB, normalize to 0~1)
    img = cv2.imread(args.input)
    if img is None:
        raise ValueError(f"无法读取图片: {args.input}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    
    # 7. 运行推理
    with torch.no_grad():
        output_tensor = model(img_tensor)
    
    # 8. 后处理 (Tensor -> numpy, * 255, clip to 0-255)
    if isinstance(output_tensor, (list, tuple)):
        output_tensor = output_tensor[0]
    
    out_img = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)
    
    # 9. 保存输出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    
    print("SUCCESS")


if __name__ == "__main__":
    main()
