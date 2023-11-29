from .runtime import CombinedExampleRT, combined_rt_name
from .sequence import combined_example_sequence, combined_seq_name

implements = {
    "runtimes": {
        combined_rt_name: {
            "build_required": True,
            "RuntimeClass": CombinedExampleRT,
            "supported_devices": {
                "x86": {},
                "example_family": {"part1": ["config1", "config2"]},
            },
            "default_sequence": combined_example_sequence,
        }
    },
    "sequences": {
        combined_seq_name: {
            "sequence_instance": combined_example_sequence,
        }
    },
}
