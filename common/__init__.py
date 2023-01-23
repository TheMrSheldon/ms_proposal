import os
from typing import Optional


def set_cuda_devices_env(devices: Optional[list[str]]):
    if devices is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print(f"Set CUDA_DEVICE_ORDER to '{os.environ['CUDA_DEVICE_ORDER']}'")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if not devices else ",".join(map(str, devices))
        print(f"Set CUDA_VISIBLE_DEVICES to '{os.environ['CUDA_VISIBLE_DEVICES']}'")


__all__ = [
    "load_model",
]
