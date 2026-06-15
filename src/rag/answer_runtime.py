from __future__ import annotations

from typing import Any

from .answer_types import Device


def _resolve_device(device: Device) -> str:
    import torch

    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for generation, but torch.cuda.is_available() is false")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _generation_torch_dtype(device: str) -> Any:
    if device != "cuda":
        return "auto"
    import torch

    major, _minor = torch.cuda.get_device_capability()
    if major < 8:
        return torch.float16
    return "auto"


def _move_inputs(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _input_length(inputs: Any) -> int:
    input_ids = inputs["input_ids"]
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    return len(input_ids[0])
