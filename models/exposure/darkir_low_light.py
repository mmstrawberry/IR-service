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
    """DarkIR 低光增强算法"""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    
    # 找权重
    model_name = options.get("model", "DarkIR_allLOL.pt")
    model_path = Path("third_party/DarkIR/weights") / model_name
    model_path = model_path.resolve()
    
    if not model_path.exists():
        model_path = Path("weights/darkir") / model_name
        model_path = model_path.resolve()
        
    if not model_path.exists():
        return {"error": f"Model weights not found: {model_path}", "status": "failed"}
        
    if not input_path.exists():
        return {"error": f"Input file not found: {input_path}", "status": "failed"}
        
    # 处理参数
    resize = options.get("resize", False)
    if isinstance(resize, str):
        resize = resize.lower() in ["true", "1", "yes"]
        
    try:
        # 调用外挂环境
        darkir_python = "/usr/local/miniconda3/envs/darkir/bin/python"
        cmd = [
            darkir_python,
            "third_party/DarkIR/bridge_infer.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--model", str(model_path),
        ]
        
        if resize:
            cmd.append("--resize")
            
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).parent.parent.parent,  # 项目根目录
        )
        
        if result.returncode != 0:
            return {"error": f"DarkIR inference failed: {result.stderr}", "status": "failed"}
            
        if not output_path.exists():
            return {"error": "Output file was not generated", "status": "failed"}
            
        return {
            "status": "success",
            "output_path": str(output_path),
            "message": "DarkIR low-light enhancement completed successfully",
        }
        
    except subprocess.TimeoutExpired:
        return {"error": "DarkIR inference timed out after 120 seconds", "status": "failed"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "status": "failed"}