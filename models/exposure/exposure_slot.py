from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from core.registry import register_algorithm


@register_algorithm(
    task="exposure_correction",
    name="exposure_slot",
    requires_gpu=True,
)
def run_exposure_slot(
    input_path: Path,
    output_path: Path,
    options: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Exposure-slot 曝光修正算法。
    
    Options expected:
      - dataset: "SICE" / "MSEC" / "LCDP" (default: SICE)
      - level: 2 或 3 (default: 2)
      - weights: 权重文件名，如 "SICE_level2.pth" (默认根据 dataset+level 自动选择)
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    
    # 从 options 获取参数
    dataset = options.get("dataset", "SICE").upper()
    level = int(options.get("level", 2))
    
    # 验证 dataset 和 level
    if dataset not in ["SICE", "MSEC", "LCDP"]:
        return {
            "error": f"Invalid dataset: {dataset}. Must be SICE, MSEC, or LCDP",
            "status": "failed",
        }
    
    if level not in [2, 3]:
        return {
            "error": f"Invalid level: {level}. Must be 2 or 3",
            "status": "failed",
        }
    
    # 自动确定权重文件路径
    weights_name = options.get("weights", f"{dataset}_level{level}.pth")
    weights_path = Path("weights/exposure-slot") / weights_name
    weights_path = weights_path.resolve()
    
    # 如果权重路径不存在，尝试备用位置
    if not weights_path.exists():
        weights_path = Path("third_party/Exposure-slot/ckpt") / weights_name
        weights_path = weights_path.resolve()
    
    # 验证权重文件存在
    if not weights_path.exists():
        return {
            "error": f"Weights file not found: {weights_path}",
            "status": "failed",
        }
    
    # 验证输入文件存在
    if not input_path.exists():
        return {
            "error": f"Input file not found: {input_path}",
            "status": "failed",
        }
    
    try:
        # # 调用 bridge_infer.py（运行在 exposure-slot 环境中）
        # cmd = [
        #     "conda",
        #     "run",
        #     "-n",
        #     "exposure-slot",
        #     "python",
        #     "third_party/Exposure-slot/bridge_infer.py",
        #     "--input",
        #     str(input_path),
        #     "--output",
        #     str(output_path),
        #     "--weights",
        #     str(weights_path),
        #     "--dataset",
        #     dataset,
        #     "--level",
        #     str(level),
        # ]
        # 抛弃 conda run，使用物理外挂环境的绝对路径
        exposureslot_python = "/usr/local/miniconda3/envs/exposureslot/bin/python"
        
        cmd = [
            exposureslot_python,
            "third_party/Exposure-slot/bridge_infer.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--weights", str(weights_path),
            "--dataset", dataset,
            "--level", str(level),
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
                "error": f"Exposure-slot inference failed: {result.stderr}",
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
            "message": "Exposure-slot exposure correction completed successfully",
            "dataset": dataset,
            "level": level,
        }
    
    except subprocess.TimeoutExpired:
        return {
            "error": "Exposure-slot inference timed out after 120 seconds",
            "status": "failed",
        }
    except Exception as e:
        return {
            "error": f"Unexpected error during Exposure-slot inference: {str(e)}",
            "status": "failed",
        }
