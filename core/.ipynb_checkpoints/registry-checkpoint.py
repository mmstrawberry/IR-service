from __future__ import annotations

from dataclasses import dataclass
import importlib
import pkgutil
from threading import RLock
from pathlib import Path
from typing import Any, Callable


AlgorithmRunner = Callable[[Path, Path, dict[str, Any]], dict[str, Any] | None]


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    task: str
    name: str
    requires_gpu: bool
    runner: AlgorithmRunner
    module: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "name": self.name,
            "requires_gpu": self.requires_gpu,
            "module": self.module,
        }


class RegistryError(Exception):
    pass


class AlgorithmAlreadyRegisteredError(RegistryError):
    pass


class AlgorithmNotFoundError(RegistryError):
    pass

# 任务 -> 算法名称 
_REGISTRY: dict[str, dict[str, AlgorithmSpec]] = {}
_IMPORTED_PACKAGES: set[str] = set()
#线程锁
_LOCK = RLock()

#注册算法的装饰器
# 当你在一个函数头上写 @register_algorithm(task="去模糊", name="DeblurDiff", requires_gpu=True) 时，这段代码就会被触发
def register_algorithm(task: str, name: str, requires_gpu: bool = False):

    if not task or not task.strip():
        raise ValueError("task must not be empty")
    if not name or not name.strip():
        raise ValueError("name must not be empty")

    normalized_task = task.strip()
    normalized_name = name.strip()

    def decorator(func: AlgorithmRunner) -> AlgorithmRunner:
        if not callable(func):
            raise TypeError("registered target must be callable")

        spec = AlgorithmSpec(
            task=normalized_task,
            name=normalized_name,
            requires_gpu=requires_gpu,
            runner=func,
            module=func.__module__,
        )

        with _LOCK:
            bucket = _REGISTRY.setdefault(normalized_task, {})
            if normalized_name in bucket:
                raise AlgorithmAlreadyRegisteredError(
                    f"Algorithm already registered: task={normalized_task}, name={normalized_name}"
                )
            bucket[normalized_name] = spec

        return func

    return decorator


# 这个函数利用了 pkgutil 和 importlib。它的逻辑是：“去 models 这个文件夹里挨个扫街（walk_packages），只要看到是 Python 文件，就自动帮我 import 进来。
def autodiscover_algorithms(package_name: str = "models") -> None:
    """
    Import all Python modules under the package once,
    enabling "drop a new file + decorate it".
    """
    with _LOCK:
        if package_name in _IMPORTED_PACKAGES:
            return

    package = importlib.import_module(package_name)
    if not hasattr(package, "__path__"):
        raise ValueError(f"{package_name} is not a package")

    for module_info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
        importlib.import_module(module_info.name)

    with _LOCK:
        _IMPORTED_PACKAGES.add(package_name)


def get_algorithm(task: str, name: str) -> AlgorithmSpec:
    try:
        return _REGISTRY[task][name]
    except KeyError as exc:
        raise AlgorithmNotFoundError(f"Algorithm not found: task={task}, name={name}") from exc


#以下三个函数是前端查询接口
def list_algorithms() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with _LOCK:
        for task_name in sorted(_REGISTRY.keys()):
            for algo_name in sorted(_REGISTRY[task_name].keys()):
                records.append(_REGISTRY[task_name][algo_name].to_metadata())
    return records


def list_algorithms_grouped() -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    with _LOCK:
        for task_name in sorted(_REGISTRY.keys()):
            algorithms = [
                _REGISTRY[task_name][algo_name].to_metadata()
                for algo_name in sorted(_REGISTRY[task_name].keys())
            ]
            grouped.append(
                {
                    "task": task_name,
                    "algorithms": algorithms,
                }
            )
    return grouped