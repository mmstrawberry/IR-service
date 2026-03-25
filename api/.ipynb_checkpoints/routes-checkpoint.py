from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from core.registry import AlgorithmNotFoundError, get_algorithm, list_algorithms_grouped

router = APIRouter()

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
WORKDIR = Path("workdirs") / "tmp"
WORKDIR.mkdir(parents=True, exist_ok=True)

# 核心修改：在注册时为每个算法指定具体路径
ALGORITHMS_PATHS = {
    "cotf_realtime_exposure": Path("/root/project/IR-service/third_party/CoTF/bridge_infer.py"),
    "darkir_realtime_exposure": Path("/root/project/IR-service/models/exposure/darkir.py"),
    # 继续添加其他算法路径
}

def _gpu_available() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return shutil.which("nvidia-smi") is not None


def _safe_error_detail(message: str, code: str) -> dict[str, str]:
    return {"code": code, "message": message}


@router.get("/algorithms")
async def get_algorithms() -> dict[str, Any]:
    tasks = list_algorithms_grouped() or []
    total_algorithms = sum(len(item.get("algorithms", [])) for item in tasks)
    return {
        "tasks": tasks,
        "total_tasks": len(tasks),
        "total_algorithms": total_algorithms,
    }


@router.post("/process")
async def process_image(
    task: str = Form(...),
    algorithm: str = Form(...),
    file: UploadFile = File(...),
    options: str | None = Form(default=None),
) -> Response:
    try:
        # 强制修复：动态加载路径，打印调试信息
        print(f"Loading algorithm: {algorithm} for task: {task}...")
        
        # 获取算法路径并调试输出
        if algorithm in ALGORITHMS_PATHS:
            algorithm_path = ALGORITHMS_PATHS[algorithm]
            print(f"Algorithm {algorithm} found at {algorithm_path}")
        else:
            raise HTTPException(
                status_code=404,
                detail=_safe_error_detail(f"Algorithm {algorithm} not found.", "ALGORITHM_NOT_FOUND"),
            )

        # 加载 CoTF 脚本
        spec = get_algorithm(task=task, name=algorithm)

        if spec.requires_gpu and not _gpu_available():
            raise HTTPException(
                status_code=400,
                detail=_safe_error_detail(
                    f"算法 `{spec.name}` 依赖 GPU / CUDA，当前环境未检测到可用 GPU。",
                    "GPU_REQUIRED",
                ),
            )

        suffix = Path(file.filename or "upload.png").suffix.lower() or ".png"
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=_safe_error_detail(
                    f"不支持的文件类型: {suffix}。仅支持 {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                    "UNSUPPORTED_FILE_TYPE",
                ),
            )

        try:
            parsed_options = json.loads(options) if options else {}
            if not isinstance(parsed_options, dict):
                raise ValueError("options must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_safe_error_detail("options 必须是合法 JSON 对象。", "INVALID_OPTIONS"),
            ) from exc

        job_dir = WORKDIR / uuid4().hex
        job_dir.mkdir(parents=True, exist_ok=True)

        input_path = job_dir / f"input{suffix}"
        output_path = job_dir / "output.png"

        try:
            input_bytes = await file.read()
            input_path.write_bytes(input_bytes)

            try:
                with Image.open(input_path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=_safe_error_detail("上传文件不是有效图像。", "INVALID_IMAGE"),
                ) from exc

            runner_result = await run_in_threadpool(
                spec.runner, input_path, output_path, parsed_options
            )

            if isinstance(runner_result, dict):
                status = str(runner_result.get("status", "")).lower()

                if status == "failed":
                    error_message = str(
                        runner_result.get("error")
                        or runner_result.get("message")
                        or "算法执行失败，但未提供详细错误信息。"
                    )
                    raise RuntimeError(error_message)

                returned_output_path = runner_result.get("output_path")
                if returned_output_path:
                    output_path = Path(returned_output_path)

            if not output_path.exists():
                raise RuntimeError(
                    f"算法执行完成，但未生成输出文件: {output_path}。请检查 wrapper 的输出逻辑。"
                )

            if output_path.stat().st_size == 0:
                raise RuntimeError(f"算法执行完成，但输出文件为空: {output_path}。")

            result_bytes = output_path.read_bytes()
            return Response(
                content=result_bytes,
                media_type="image/png",
                headers={
                    "X-Task": spec.task,
                    "X-Algorithm": spec.name,
                    "X-Requires-GPU": str(spec.requires_gpu).lower(),
                },
            )

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=_safe_error_detail(f"处理失败: {exc}", "PROCESSING_FAILED"),
            ) from exc
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=_safe_error_detail(f"处理失败: {exc}", "PROCESSING_FAILED"),
        ) from exc