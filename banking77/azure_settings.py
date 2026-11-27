"""Shared Azure job identity and explicit device/build choices."""


def job_name(launch_id: str, index: int, *, attempt: int = 0) -> str:
    name = f"banking77-{launch_id}-{index}"
    return f"{name}-r{attempt}" if attempt else name


def torch_device(azure_device: str) -> str:
    return {"cpu": "cpu", "gpu": "cuda"}[azure_device]


def torch_build(device: str) -> str:
    return {"cpu": "cpu", "cuda": "cu126"}[device]
