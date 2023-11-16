import turnkeyml.build.sequences as sequences
from .runtime import TensorRT


implements = {
    "runtimes": {
        "trt": {
            "build_required": True,
            "RuntimeClass": TensorRT,
            "supported_devices": {"nvidia"},
            "default_sequence": sequences.optimize_fp16,
        }
    }
}
