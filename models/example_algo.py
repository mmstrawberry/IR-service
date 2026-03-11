from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance

from core.registry import register_algorithm


@register_algorithm(task="exposure_correction", name="example_gamma_boost", requires_gpu=False)
def run_example_gamma_boost(
    input_path: Path,
    output_path: Path,
    options: dict[str, Any],
) -> dict[str, Any]:
    """
    示例算法模板。
    真实算法接入时，把这里替换成你对 third_party/ 中开源仓库的调用。
    """
    brightness = float(options.get("brightness", 1.2))

    with Image.open(input_path).convert("RGB") as image:
        result = ImageEnhance.Brightness(image).enhance(brightness)
        result.save(output_path, format="PNG")

    return {
        "message": "example algorithm completed",
        "output_path": str(output_path),
        "brightness": brightness,
    }