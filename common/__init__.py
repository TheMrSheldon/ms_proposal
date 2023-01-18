import os


def set_cuda_devices_env(devices: list[str]):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    print(f"Set CUDA_DEVICE_ORDER to '{os.environ['CUDA_DEVICE_ORDER']}'")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if not devices else ",".join(map(str, devices))
    print(f"Set CUDA_VISIBLE_DEVICES to '{os.environ['CUDA_VISIBLE_DEVICES']}'")


__all__ = [
    "load_model",
]
