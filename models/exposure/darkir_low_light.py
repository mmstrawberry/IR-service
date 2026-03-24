from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from core.registry import register_algorithm


@register_algorithm(
    task="exposure_correction",
    name="darkir_low_light",
    requires_gpu=True,
)
def run_darkir_low_light(
    input_path: Path,
    output_path: Path,
    options: dict[str, Any],
) -> dict[str, Any] | None:
    """
    DarkIR 低光增强算法（去欠曝）。
    同一个 DarkIR 模型也支持去模糊任务。
    
    Options expected:
      - model: 模型权重路径 (default: third_party/DarkIR/weights/DarkIR_LOLBlur.pth)
      - resize: 是否对大图片进行缩小处理 (default: False)
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    
    # 从 options 获取模型路径
    model_name = options.get("model", "DarkIR_LOLBlur.pth")
    model_path = Path("third_party/DarkIR/weights") / model_name
    model_path = model_path.resolve()
    
    # 如果权重路径不存在，尝试备用位置
    if not model_path.exists():
        model_path = Path("weights/darkir") / model_name
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
    
    # 是否启用缩放
    resize = options.get("resize", False)
    if isinstance(resize, str):
        resize = resize.lower() in ["true", "1", "yes"]
    
    try:
        # 调用 bridge_infer.py（运行在 darkir 环境中）
        cmd = [
            "conda",
            "run",
            "-n",
            "darkir",
            "python",
            "third_party/DarkIR/bridge_infer.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            str(model_path),
            "--resize",
            str(resize),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).parent.parent.parent,  # 项目根目录
        )
        
        if result.returncode != 0:
            return {
                "error": f"DarkIR low-light inference failed: {result.stderr}",
                "status": "failed",
            }
        
        # 验证输出文件是否存在
        if not output_path.exists():
            return {
                "error": "Output file was not generated",
                "status": "failed",
            }
        
        return {
            "status": "success",
            "output_path": str(output_path),
            "message": "DarkIR low-light enhancement completed successfully",
        }
    
    except subprocess.TimeoutExpired:
        return {
            "error": "DarkIR inference timed out after 120 seconds",
            "status": "failed",
        }
    except Exception as e:
        return {
            "error": f"Unexpected error during DarkIR inference: {str(e)}",
            "status": "failed",
        }
