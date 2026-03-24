from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from core.registry import register_algorithm


@register_algorithm(
    task="exposure_correction",
    name="cotf_realtime_exposure",
    requires_gpu=True,
)
def run_cotf(
    input_path: Path,
    output_path: Path,
    options: dict[str, Any],
) -> dict[str, Any] | None:
    """
    CoTF (Content-to-Film Transfer) exposure correction algorithm.
    
    Options expected:
      - weights: Path to model weights (default: weights/cotf/sice_net.pth)
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    
    # 从 options 中获取权重路径，默认使用 sice_net.pth
    weights_name = options.get("weights", "sice_net.pth")
    weights_path = Path("weights/cotf") / weights_name
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
        # 调用 bridge_infer.py，运行在 cotf 环境中
        cmd = [
            "conda",
            "run",
            "-n",
            "cotf",
            "python",
            "third_party/CoTF/bridge_infer.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--weights",
            str(weights_path),
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
                "error": f"CoTF inference failed: {result.stderr}",
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
            "message": "CoTF exposure correction completed successfully",
        }
    
    except subprocess.TimeoutExpired:
        return {
            "error": "CoTF inference timed out after 120 seconds",
            "status": "failed",
        }
    except Exception as e:
        return {
            "error": f"Unexpected error during CoTF inference: {str(e)}",
            "status": "failed",
        }
