"""Inspect local hardware and pick a model profile. Run: python -m saree_ai.hardware"""
from __future__ import annotations

import json
import os
import platform

MIN_RAM_GB = float(os.environ.get("MIN_RAM_GB", "4.0"))     # overridable for small cloud instances
MIN_CPU_CORES = int(os.environ.get("MIN_CPU_CORES", "4"))


def inspect() -> dict:
    info: dict = {"platform": platform.platform(), "python": platform.python_version(),
                  "cpu": platform.processor(), "cpu_logical": os.cpu_count()}
    try:
        import psutil
        info["cpu_physical"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["ram_available_gb"] = round(psutil.virtual_memory().available / 2**30, 1)
    except ImportError:
        info["ram_gb"] = None
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
    except ImportError:
        info["onnxruntime"] = None
        info["ort_providers"] = []
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            p = torch.cuda.get_device_properties(0)
            info["gpu"] = p.name
            info["gpu_mem_gb"] = round(p.total_memory / 2**30, 1)
    except ImportError:
        info["torch"] = None
        info["cuda"] = False
    return info


def choose_profile(info: dict) -> dict:
    """'full' = GPU pose baseline + LR-ASPP segmentation; 'lite' = CPU everything at lower res."""
    problems = []
    if info.get("ram_gb") is not None and info["ram_gb"] < MIN_RAM_GB:
        problems.append(f"Only {info['ram_gb']} GB RAM; at least {MIN_RAM_GB} GB is required.")
    if (info.get("cpu_logical") or 0) < MIN_CPU_CORES:
        problems.append(f"Only {info.get('cpu_logical')} CPU threads; at least {MIN_CPU_CORES} are required.")
    if not info.get("onnxruntime"):
        problems.append("onnxruntime is not installed: run `pip install -r requirements.txt`.")
    gpu = bool(info.get("cuda")) and (info.get("gpu_mem_gb") or 0) >= 3.5
    profile = "full" if gpu else "lite"
    return {
        "profile": profile,
        "pose_device": "cuda" if gpu else "cpu",
        "pose_input_long_side": 640 if gpu else 384,
        "target_fps": 15 if gpu else 5,
        "meets_minimum": not problems,
        "problems": problems,
    }


if __name__ == "__main__":
    i = inspect()
    print(json.dumps({"hardware": i, "selection": choose_profile(i)}, indent=2))
