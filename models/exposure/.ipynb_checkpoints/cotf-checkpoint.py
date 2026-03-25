from __future__ import annotations

import os
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
    project_root = Path(__file__).resolve().parent.parent
    third_party_root = project_root / "third_party" / "CoTF"
    bridge_script = third_party_root / "bridge_infer.py"

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conda_env = str(options.get("conda_env", "cotf"))
    timeout = int(options.get("timeout", 300))

    # 支持两种写法：
    # 1) {"weights": "sice_net.pth"}
    # 2) {"weights": "/absolute/path/to/sice_net.pth"}
    weights_opt = str(options.get("weights", "sice_net.pth"))
    weights_candidate = Path(weights_opt)
    if weights_candidate.is_absolute():
        weights_path = weights_candidate.resolve()
    else:
        weights_path = (third_party_root / "weights" / weights_opt).resolve()

    debug_log = output_path.parent / "cotf_wrapper_debug.log"

    def write_debug(text: str) -> None:
        try:
            with open(debug_log, "a", encoding="utf-8") as f:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")
        except Exception:
            pass

    write_debug("==== CoTF wrapper start ====")
    write_debug(f"project_root = {project_root}")
    write_debug(f"third_party_root = {third_party_root}")
    write_debug(f"bridge_script = {bridge_script}")
    write_debug(f"input_path = {input_path}")
    write_debug(f"output_path = {output_path}")
    write_debug(f"weights_path = {weights_path}")
    write_debug(f"conda_env = {conda_env}")
    write_debug(f"timeout = {timeout}")

    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        write_debug(msg)
        return {"status": "failed", "error": msg}

    if not third_party_root.exists():
        msg = f"CoTF directory not found: {third_party_root}"
        write_debug(msg)
        return {"status": "failed", "error": msg}

    if not bridge_script.exists():
        msg = f"bridge_infer.py not found: {bridge_script}"
        write_debug(msg)
        return {"status": "failed", "error": msg}

    if not weights_path.exists():
        msg = f"Weights file not found: {weights_path}"
        write_debug(msg)
        return {"status": "failed", "error": msg}

    conda_exe = shutil.which("conda")
    if conda_exe is None:
        # 常见 conda 安装路径兜底
        candidates = [
            "/root/miniconda/bin/conda",
            "/root/miniconda3/bin/conda",
            "/root/anaconda3/bin/conda",
            "/opt/conda/bin/conda",
        ]
        for c in candidates:
            if Path(c).exists():
                conda_exe = c
                break

    if conda_exe is None:
        msg = "conda executable not found in PATH"
        write_debug(msg)
        return {"status": "failed", "error": msg}

    cmd = [
        conda_exe,
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

    write_debug("command = " + " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_root),
            env=env,
        )
    except subprocess.TimeoutExpired:
        msg = f"CoTF inference timed out after {timeout} seconds"
        write_debug(msg)
        return {"status": "failed", "error": msg}
    except Exception as e:
        msg = f"Unexpected error while launching CoTF: {e}"
        write_debug(msg)
        return {"status": "failed", "error": msg}

    write_debug(f"returncode = {result.returncode}")
    write_debug("----- STDOUT -----")
    write_debug(result.stdout or "")
    write_debug("----- STDERR -----")
    write_debug(result.stderr or "")

    if result.returncode != 0:
        return {
            "status": "failed",
            "error": (
                f"CoTF inference failed with exit code {result.returncode}\n"
                f"Command: {' '.join(cmd)}\n"
                f"See debug log: {debug_log}"
            ),
        }

    if output_path.exists() and output_path.is_file() and output_path.stat().st_size > 0:
        write_debug("output file exists and is non-empty")
        return {
            "status": "success",
            "output_path": str(output_path),
            "message": "CoTF exposure correction completed successfully",
        }

    # 再次兜底：有些脚本可能稍晚落盘，短暂轮询一下
    import time

    for _ in range(10):
        time.sleep(0.3)
        if output_path.exists() and output_path.is_file() and output_path.stat().st_size > 0:
            write_debug("output file appeared during retry wait")
            return {
                "status": "success",
                "output_path": str(output_path),
                "message": "CoTF exposure correction completed successfully",
            }

    write_debug("output file still missing after subprocess success")
    return {
        "status": "failed",
        "error": (
            "CoTF finished but output file was not generated.\n"
            f"Expected output: {output_path}\n"
            f"See debug log: {debug_log}"
        ),
    }