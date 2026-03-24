from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from core.registry import AlgorithmNotFoundError, get_algorithm, list_algorithms_grouped

router = APIRouter()

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
WORKDIR = Path("workdirs") / "tmp"
WORKDIR.mkdir(parents=True, exist_ok=True)


def _gpu_available() -> bool:
    """
    尽力检测 GPU 是否可用。
    优先使用 torch.cuda.is_available()，失败则退化到 nvidia-smi 探测。
    """
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return shutil.which("nvidia-smi") is not None


def _safe_error_detail(message: str, code: str) -> dict[str, str]:
    return {"code": code, "message": message}

# 列出所有可用的算法
@router.get("/algorithms")
async def get_algorithms() -> dict[str, Any]:
    tasks = list_algorithms_grouped() or []
    total_algorithms = sum(len(item.get("algorithms", [])) for item in tasks)
    return {
        "tasks": tasks,
        "total_tasks": len(tasks),
        "total_algorithms": total_algorithms,
    }


# 接收前端上传的图片文件、任务类型、算法名称等参数 → 校验参数合法性 → 调用指定算法处理图片 
# → 返回处理后的图片结果 → 最后清理临时文件。
#图片处理后端接口
@router.post("/process")

async def process_image(
    task: str = Form(...),
    algorithm: str = Form(...),
    file: UploadFile = File(...),
    options: str | None = Form(default=None),
) -> Response:
    try:
        spec = get_algorithm(task=task, name=algorithm)
    except AlgorithmNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=_safe_error_detail(str(exc), "ALGORITHM_NOT_FOUND"),
        ) from exc

    if spec.requires_gpu and not _gpu_available():
        raise HTTPException(
            status_code=400,
            detail=_safe_error_detail(
                f"算法 `{spec.name}` 依赖 GPU / CUDA 自定义算子，当前运行环境未检测到可用 GPU。",
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

        await run_in_threadpool(spec.runner, input_path, output_path, parsed_options)

        if not output_path.exists():
            raise RuntimeError(
                f"算法执行完成，但未生成输出文件: {output_path}。请检查 wrapper 的输出逻辑。"
            )

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