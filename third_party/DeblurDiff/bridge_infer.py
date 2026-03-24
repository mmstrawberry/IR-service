import argparse
import os
import sys
import subprocess
from pathlib import Path

# 防暴毙机制
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入图片路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--steps", type=int, default=50, help="推理步数")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--tile_size", type=int, default=None, help="分块大小")
    parser.add_argument("--tile_stride", type=int, default=None, help="分块步长")
    args = parser.parse_args()
    
    # 1. 验证输入文件
    if not os.path.exists(args.input):
        raise ValueError(f"无法读取输入图片: {args.input}")
    
    if not os.path.exists(args.model):
        raise ValueError(f"无法读取模型权重: {args.model}")
    
    # 2. 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. 构建 inference.py 命令
    # 注意：inference.py 在当前环境中已安装，不再需要切环境
    cmd = [
        "python",
        "inference.py",
        "--input",
        args.input,
        "--output",
        args.output,
        "--model",
        args.model,
        "--steps",
        str(args.steps),
        "--device",
        args.device,
    ]
    
    # 4. 【关键】只有当 tile_size 和 tile_stride 都被指定时，才启用 tiled
    if args.tile_size is not None and args.tile_stride is not None:
        cmd.extend([
            "--tiled",
            "--tile_size",
            str(args.tile_size),
            "--tile_stride",
            str(args.tile_stride),
        ])
    
    # 5. 切换到 DeblurDiff 目录并运行 inference.py
    deblurdiff_dir = Path(__file__).parent
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(deblurdiff_dir),
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"inference.py 执行失败: {result.stderr}")
        
        print("SUCCESS")
    
    except subprocess.TimeoutExpired:
        raise RuntimeError("DeblurDiff 推理超时（>300秒）")
    except Exception as e:
        raise RuntimeError(f"DeblurDiff 推理异常: {str(e)}")


if __name__ == "__main__":
    main()
