from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from core.registry import register_algorithm


@register_algorithm(
    task="deblurring",
    name="deblur_diff",
    requires_gpu=True,
)
def run_deblur_diff(
    input_path: Path,
    output_path: Path,
    options: dict[str, Any],
) -> dict[str, Any] | None:
    """
    DeblurDiff 去模糊算法。
    
    Options expected:
      - model: 模型权重路径 (default: third_party/DeblurDiff/weights/model.pth)
      - steps: 推理步数 (default: 50)
      - device: 计算设备 (default: cuda)
      - tile_size: 分块大小（可选，只有同时指定 tile_size 和 tile_stride 才启用分块）
      - tile_stride: 分块步长（可选）
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    
    # 输出路径应该是目录，不是文件
    # 如果用户传了文件路径，转更为目录
    if str(output_path).lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
        output_dir = output_path.parent
    else:
        output_dir = output_path
    
    # 从 options 获取参数
    model_name = options.get("model", "model.pth")
    model_path = Path("third_party/DeblurDiff/weights") / model_name
    model_path = model_path.resolve()
    
    # 如果权重路径不存在，尝试备用位置
    if not model_path.exists():
        model_path = Path("weights/deblurdiff") / model_name
        model_path = model_path.resolve()
    
    # 验证权重文件存在
    if not model_path.exists():
        return {
            "error": f"Model weights not found: {model_path}",
            "status": "failed",
        }
    
    # 验证输入文件存在
    if not input_path.exists():
        return {
            "error": f"Input file not found: {input_path}",
            "status": "failed",
        }
    
    # 获取推理参数
    steps = int(options.get("steps", 50))
    device = options.get("device", "cuda")
    tile_size = options.get("tile_size")
    tile_stride = options.get("tile_stride")
    
    # 尝试转换 tile 参数为整数（如果提供了）
    if tile_size is not None:
        try:
            tile_size = int(tile_size)
        except (ValueError, TypeError):
            tile_size = None
    
    if tile_stride is not None:
        try:
            tile_stride = int(tile_stride)
        except (ValueError, TypeError):
            tile_stride = None
    
    try:
        # 【修改 1】使用绝对路径，彻底抛弃 conda run！
        deblurdiff_python = "/usr/local/miniconda3/envs/deblurdiff/bin/python"
        
        cmd = [
            deblurdiff_python,
            "third_party/DeblurDiff/bridge_infer.py",
            "--input", str(input_path),
            "--output", str(output_dir),  # 传给 bridge 的依然是目录
            "--model", str(model_path),
            "--steps", str(steps),
            "--device", device,
        ]
        # 【关键】只有当 tile_size 和 tile_stride 都被指定时，才启用 tiled
        if tile_size is not None and tile_stride is not None:
            cmd.extend([
                "--tile_size",
                str(tile_size),
                "--tile_stride",
                str(tile_stride),
            ])
    
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(__file__).parent.parent.parent,  # 项目根目录
        )
    
        if result.returncode != 0:
            return {
                "error": f"DeblurDiff inference failed: {result.stderr}",
                "status": "failed",
            }
        
        # DeblurDiff 的 inference.py 会在 output_dir 中生成文件
        # 查找生成的输出文件
        output_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
        
        if not output_files:
            return {
                "error": "No output file was generated in output directory",
                "status": "failed",
            }
        
        # 返回第一个生成的文件作为输出
        generated_output = output_files[0]
        
        return {
            "status": "success",
            "output_path": str(generated_output),
            "message": "DeblurDiff deblurring completed successfully",
            "steps": steps,
            "tiled": tile_size is not None and tile_stride is not None,
        }
    
    except subprocess.TimeoutExpired:
        return {
            "error": "DeblurDiff inference timed out after 300 seconds",
            "status": "failed",
        }
    except Exception as e:
        return {
            "error": f"Unexpected error during DeblurDiff inference: {str(e)}",
            "status": "failed",
        }