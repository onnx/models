from .runtime import TorchRT

implements = {
    "runtimes": {
        "torch-eager": {
            "build_required": False,
            "RuntimeClass": TorchRT,
            "supported_devices": {"x86"},
        },
        "torch-compiled": {
            "build_required": False,
            "RuntimeClass": TorchRT,
            "supported_devices": {"x86"},
        },
    }
}
