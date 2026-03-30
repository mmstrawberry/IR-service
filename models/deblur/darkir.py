from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from core.registry import register_algorithm


@register_algorithm(
    task="deblurring",
    name="darkir",
    requires_gpu=True,
)
def run_darkir_deblur(
    input_path: Path,
    output_path: Path,
    options: dict[str, Any],
) -> dict[str, Any] | None:
    """
    DarkIR 去模糊算法（处理低光模糊图）。
    同一个 DarkIR 模型也支持低光增强任务。
    
    Options expected:
      - model: 模型权重路径 (default: third_party/DarkIR/weights/DarkIR_LOLBlur.pth)
      - resize: 是否对大图片进行缩小处理 (default: False)
    """
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    
    # 从 options 获取模型路径
    model_name = options.get("model", "DarkIR_384.pt")
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
    
    # try:
    #     # 调用 bridge_infer.py（运行在 darkir 环境中）
    #     cmd = [
    #         "conda",
    #         "run",
    #         "-n",
    #         "darkir",
    #         "python",
    #         "third_party/DarkIR/bridge_infer.py",
    #         "--input",
    #         str(input_path),
    #         "--output",
    #         str(output_path),
    #         "--model",
    #         str(model_path),
    #         "--resize",
    #         str(resize),
    #     ]
        # ... 前面的验证代码保持不变 ...

    # 是否启用缩放
    resize = options.get("resize", False)
    if isinstance(resize, str):
        resize = resize.lower() in ["true", "1", "yes"]
    
    try:
        # 【修改 1】使用绝对路径，彻底抛弃 conda run，防止服务卡死！
        darkir_python = "/usr/local/miniconda3/envs/darkir/bin/python"
        
        # 基础命令组装（不含 resize）
        cmd = [
            darkir_python,
            "third_party/DarkIR/bridge_infer.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--model", str(model_path),
        ]
        
        # 【修改 2】因为桥接脚本里用的是 action="store_true"，所以只有为 True 时才追加这个标志
        if resize:
            cmd.append("--resize")
            
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).parent.parent.parent,  # 项目根目录
        )
        
        # ... 后面的 result 校验代码保持不变 ...
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).parent.parent.parent,  # 项目根目录
        )
        
        if result.returncode != 0:
            return {
                "error": f"DarkIR deblur inference failed: {result.stderr}",
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
            "message": "DarkIR deblurring completed successfully",
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
