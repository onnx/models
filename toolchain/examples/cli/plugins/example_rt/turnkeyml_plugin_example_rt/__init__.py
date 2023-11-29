from .runtime import ExampleRT, example_rt_name

implements = {
    "runtimes": {
        example_rt_name: {
            "build_required": False,
            "RuntimeClass": ExampleRT,
            "supported_devices": {"x86"},
            # magic_perf_points and super_runtime_points are custom stats we will
            # have to set in the ExampleRT.
            "status_stats": ["magic_perf_points", "super_runtime_points"],
        }
    }
}
