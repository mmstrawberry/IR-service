from __future__ import annotations

import shutil
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

    Options:
      - weights: 权重文件名，默认 sice_net.pth
      - conda_env: conda 环境名，默认 cotf
      - timeout: 超时时间，默认 300 秒
    """
    project_root = Path(__file__).resolve().parent.parent
    third_party_root = project_root / "third_party" / "CoTF"
    bridge_script = third_party_root / "bridge_infer.py"

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conda_env = options.get("conda_env", "cotf")
    timeout = int(options.get("timeout", 300))

    weights_name = options.get("weights", "sice_net.pth")
    weights_path = (project_root / "weights" / "cotf" / weights_name).resolve()

    # 基础检查
    if not input_path.exists():
        return {
            "status": "failed",
            "error": f"Input file not found: {input_path}",
        }

    if not bridge_script.exists():
        return {
            "status": "failed",
            "error": f"bridge_infer.py not found: {bridge_script}",
        }

    if not weights_path.exists():
        return {
            "status": "failed",
            "error": f"Weights file not found: {weights_path}",
        }

    # 记录运行前 results 目录中已有图片，方便后面做兜底查找
    results_dir = third_party_root / "results"
    before_files = set()
    if results_dir.exists():
        before_files = {
            p.resolve()
            for p in results_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        }

    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(bridge_script),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--weights",
        str(weights_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_root),
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "error": f"CoTF inference timed out after {timeout} seconds",
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Unexpected error during CoTF inference: {e}",
        }

    # 子进程失败
    if result.returncode != 0:
        return {
            "status": "failed",
            "error": (
                f"CoTF inference failed with code {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            ),
        }

    # 1) 优先检查 bridge_infer 是否已按要求输出到 output_path
    if output_path.exists() and output_path.is_file() and output_path.stat().st_size > 0:
        return {
            "status": "success",
            "output_path": str(output_path),
            "message": "CoTF exposure correction completed successfully",
            "stdout": result.stdout[-2000:],
        }

    # 2) 兜底：如果 bridge_infer 没写到 output_path，尝试从 results 目录找新生成的图片
    after_files = set()
    if results_dir.exists():
        after_files = {
            p.resolve()
            for p in results_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        }

    new_files = list(after_files - before_files)

    if new_files:
        # 按修改时间排序，取最新的那个
        new_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_file = new_files[0]
        try:
            shutil.copy2(latest_file, output_path)
            if output_path.exists() and output_path.stat().st_size > 0:
                return {
                    "status": "success",
                    "output_path": str(output_path),
                    "message": f"CoTF output recovered from results dir: {latest_file}",
                    "stdout": result.stdout[-2000:],
                }
        except Exception as e:
            return {
                "status": "failed",
                "error": (
                    f"CoTF finished but failed to copy fallback result.\n"
                    f"Fallback file: {latest_file}\n"
                    f"Copy error: {e}\n"
                    f"STDOUT:\n{result.stdout}\n"
                    f"STDERR:\n{result.stderr}"
                ),
            }

    # 3) 最终失败：没有找到输出
    return {
        "status": "failed",
        "error": (
            "CoTF finished but no output file was generated.\n"
            f"Expected output: {output_path}\n"
            f"Checked fallback dir: {results_dir}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        ),
    }