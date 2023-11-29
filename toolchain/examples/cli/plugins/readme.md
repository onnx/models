# CLI Example Plugins

This directory contains plugins that can be installed to demonstrate how turnkeyml can be extended via the plugin interface:
- `example_rt`: Example of a runtime plugin. Install with `pip install -e example_rt` to add the `example-rt` runtime to your turnkey CLI.
- `example_seq`: Example of a sequence plugin. Install with `pip install -e example_seq` to add the `example-seq` sequence to your turnkey CLI.
- `example_combined`: Example of a plugin that includes both a sequence and a runtime. Install with `pip install -e example_combined` to add the `example-combined-rt` runtime and `example-combined-seq` sequence to your turnkey CLI.

See the [Tools User Guide plugins section](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#plugins) for information about how to create plugins.