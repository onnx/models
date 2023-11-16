# TurnkeyML Contribution Guide

Hello and welcome to the project! 🎉

We're thrilled that you're considering contributing to the project. This project is a collaborative effort and we welcome contributors from everyone.

Before you start, please take a few moments to read through these guidelines. They are designed to make the contribution process easy and effective for everyone involved. Also take a look at the [code organization](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/code.md) for a bird's eye view of the repository.

The guidelines document is organized as the following sections:
- [Contributing a model](#contributing-a-model)
- [Contributing a plugin](#contributing-a-plugin)
- [Contributing to the overall framework](#contributing-to-the-overall-framework)
- [Issues](#issues)
- [Pull Requests](#pull-requests)
- [Testing](#testing)
- [Versioning](#versioning)


## Contributing a model

One of the easiest ways to contribute is to add a model to the benchmark. To do so, simply add a `.py` file to the `models/` directory that instantiates and calls a supported type of model (see [Tools User Guide](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md) to learn more). The automated benchmarking infrastructure will do the rest!

## Contributing a plugin

TurnkeyML supports a variety of built-in build sequences, runtimes, and devices (see the [Devices and Runtimes table](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#devices-runtimes-table)). You can contribute a plugin to add support for a different build sequence, runtime, or device of your choosing.

A turnkey plugin is a pip-installable package that implements support for building a model using a custom sequence and/or benchmarking a model on a [device](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#device) with a [runtime](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#runtime). These packages must adhere to a specific interface that is documented below. 

### Plugin Directory Layout

A plugin package that instantiates all of the optional files would have this directory layout:

```
`<descriptive_name>`/
        |- setup.py
        |- turnkeyml_runtime_<descriptive_name>/
                |- __init__.py
                |- sequence.py
                |- runtime.py
                |- execute.py
                |- within_conda.py
                    
```
### Package Template

Plugins are pip-installable packages, so they each take the form of a directory that contains a `setup.py` script and a Python module containing the plugin source code.

We require the following naming scheme:

- The top level directory can be named any `<descriptive_name>` you like.
  - For example, `example_rt/`
- The package name is `turnkeyml_plugin_<descriptive_name>`
  - For example, `turnkeyml_plugin_example_rt`
  - Note: this naming convention is used by the tools to detect installed plugins. If you do not follow the convention your plugin will not be detected.
- Within the module a `turnkeyml_plugin_<descriptive_name>/__init__.py` file that has an `implements` nested dictionary (see [Implements Dictionary](#implements-dictionary)).
  - Note: a single plugin can implement any number of `runtimes` and `sequences`.
- Source code files that implement the plugin capabilities (see [Plugin Directory Layout](#plugin-directory-layout)).

### Runtime Plugins

Plugins can implement one or more runtimes. 

> See [example_rt](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/examples/cli/plugins/example_rt) for an example of a minimal runtime plug in. This example is used below to help explain the interface.

To add a runtime to a plugin:

1. Pick a unique name, `<runtime_name>` for each runtime that will be supported by the plugin.
    - This name will be used in the `turnkey --runtime <runtime_name>` [argument](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#runtimes)
    - For example, a runtime named `example-rt` would be invoked with `turnkey --runtime example-rt`

1. Populate the [Implements Dictionary](#implements-dictionary) with a per-runtime dictionary with the following fields:
    - `supported_devices: Union[Set,Dict]`: combination of devices supported by the runtime.
      - For example, in `example_rt`, `"supported_devices": {"x86"}` indicates that the `x86` device is supported by the `example` runtime.
      - A `device` typically refers to an entire family of devices, such as the set of all `x86` CPUs. However, plugins can provide explicit support for specific `device parts` within a device family. Additionally, specific `configurations` within a device model (e.g., a specific device firmware) are also supported.
        - Each supported part within a device family must be defined as a dictionary.
        - Each supported configuration within a device model must be defined as a list.
        - Example: `"supported_devices": {"family1":{"part1":["config1","config2"]}}`.
        - See [example_combined](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/examples/cli/plugins/example_combined) for a plugin implementation example that leverages this feature.
      - Note: If a device is already supported by the tools, this simply adds support for another runtime to that device. If the device is _not_  already supported by the tools, this also adds support for that device and it will start to appear as an option for the `turnkey --device  <device_name>` argument.
    - `"build_required": Bool`: indicates whether the `build_model()` API should be called on the `model` and `inputs`.
    - `"docker_required": Bool`: indicates whether benchmarking is implemented through a docker container.
      - For example, `"build_required": False` indicates that no build is required, and benchmarking can be performed directly on the `model`   and `inputs`.
      - An example where `"build_required": True` is the `ort` runtime, which requires the `model` to be [built](#build) (via ONNX exporter)  into a `.onnx` file prior to benchmarking.
    - (Optional) `"default_sequence": = <instance of Sequence>`: if a build is required, this is the sequence of model-to-model transformations that the runtime expects.
      - For example, `ort` expects an ONNX file that has been optimized and converted to fp16, so it uses the built-in `sequences.onnx_fp32`  sequence.
      - If a build is not required this field can be omitted. 
    - `"RuntimeClass": <class_name>`, where `<class_name>` is a unique name for a Python class that inherits `BaseRT` and implements the runtime.
      - For example, `"RuntimeClass": ExampleRT` implements the `example` runtime.
      - The interface for the runtime class is defined in [Runtime Class](#runtime-class) below.
    - (Optional) `"status_stats": List[str]`: a list of keys from the build stats that should be printed out at the end of benchmarking in the CLI's `Status` output. These keys, and corresponding values, must be set in the runtime class using `self.stats.add_build_stat(key, value)`.
    - (Optional) `"requirement_check": Callable`: a callable that runs before each benchmark. This may be used to check whether the device selected is available and functional before each benchmarking run. Exceptions raised during this callable will halt the benchmark of all selected files.

1. Populate the package with the following files (see [Plugin Directory Layout](#plugin-directory-layout)):
    - A `runtime.py` script that implements the [Runtime Class](#runtime-class).
    - (Optional) An `execute` method that follows the [Execute Method](#execute-method) template and implements the benchmarking methodology for the device/runtime combination(s).
      - See the `tensorrt` runtime's `execute.py::main()` for a fairly minimal example.
    - (Optional) A `within_conda.py` script that executes inside the conda env to collect benchmarking results.
      - See the `onnxrt` runtime's `within_conda.py` for an example.

### Sequence Plugins

Plugins can implement one or more build sequences. 

> See [example_seq](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/examples/cli/plugins/example_seq) for an example of a minimal sequence plug in. This example is used below to help explain the interface.

To add a build sequence to a plugin:

1. Pick a unique name, `<sequence_name>` for each sequence that will be supported by the plugin.
    - This name will be used in the `turnkey --sequence <sequence_name>` [argument](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#sequence)
    - For example, a sequence named `example-seq` would be invoked with `turnkey --sequence example-seq`

1. Populate the [Implements Dictionary](#implements-dictionary) with a per-sequence dictionary with the following fields:
    - `"sequence_instance" : <instance of Sequence>`: points to an instance of `turnkeyml.build.stage.Sequence` that implements the model-to-model transformations for the build sequence. 

1. Populate the package with the following files (see [Plugin Directory Layout](#plugin-directory-layout)):
    - A `sequence.py` script that implements the Sequence Class and associated sequence instance.

### Implements Dictionary

This dictionary has keys for each type of plugin that will be installed by this package. 
- Packages with runtime plugin(s) should have a `runtimes` key in the `implements` dictionary, which in turn includes one dictionary per runtime installed in the plugin.
- Packages with sequence plugin(s) should have a `sequences` key in the `implements` dictionary, which in turn includes one dictionary per runtime installed in the plugin.

An `implements` dictionary with both sequences and runtimes would have the form:

```python
implements = {
  "runtimes": {
    "runtime_1_name" : {
      "build_required": Bool,
      "RuntimeClass": Class(BaseRT),
      "devices": List[str],
      "default_sequence": Sequence instance,
      "status_stats": ["custom_stat_1", "custom_stat_2"],
    },
    "runtime_2_name" : {...},
    ...
  }
  "sequences": {
    "sequence_name_1": {"sequence_instance": Sequence instance,},
    "sequence_name_2": {...},
    ...
  }
}
```


### Runtime Class

A runtime class inherits the abstract base class [`BaseRT`](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/src/turnkeyml/run/basert.py) and implements a one or more [runtimes](#runtime) to provide benchmarking support for one or more [devices](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md#devices). 

`BaseRT` has 4 methods that plugin developers must overload: 
- `_setup()`: any code that should be called prior to benchmarking as a one-time setup. Called automatically at the end of  `BaseRT.__init__()`.
- `mean_latency()`: returns the mean latency, in ms, for the benchmarking run.
- `throughput()`: returns the throughput, in IPS, for the benchmarking run.
- `device_name()`: returns the full device name for the device used in benchmarking. For example, a benchmark on a `x86` device might have a device name like `AMD Ryzen 7 PRO 6850U with Radeon Graphics`.
- [Optional] `_execute()`: method that `BaseRT` can automatically call during `BaseRT.benchmark()`, which implements the specific benchmarking methodology for that device/runtime combination. See [Execute Method](#execute-method) for more details.
- [Optional] `__init__()`: the `__init__` method can be overloaded to take additional keyword arguments, see [Custom Runtime Arguments](#custom-runtime-arguments) for details.

Developers may also choose to overload the `benchmark()` function. By default, `BaseRT` will automatically invoke the module's [Execute Method](#execute-method) and use `mean_latency()`, `throughput()`, and `device_name()` to populate a `MeasuredPerformance` instance to return. However, some benchmarking methodologies may not lend themselves to a dedicated execute method. For example, `TorchRT` simply implements all of its benchmarking logic within an overloaded `benchmark()` method. 

### Custom Runtime Arguments

The `turnkey` CLI/APIs allow users to pass arbitrary arguments to the runtime with `--rt-args`.

Runtime arguments from the user's `--rt-args` will be passed into the runtime class's `__init__()` method as keyword arguments. Runtime plugins must accept any such custom arguments in their overloaded `__init__()` method, at which point the contributor is free to use them any way they like. A common usage would be to store arguments as members of `self` and then access them during `_setup()` or `_execute()`.

The format for runtime arguments passed through the CLI is:

```
--rt-args arg1::value1 arg2::[value2,value3] flag_arg
```

Where:
- Arguments are space-delimited.
- Flag arguments (in the style of `argparse`'s `store_true`) are passed by key name only and result in `<key>=True`.
- Arguments with a single value are passed as `key::value`.
- Arguments that are a list of values are passed as `key::[value1, value2, ...]`.

API users can pass an arbitrary dictionary of arguments, e.g., `benchmark_files(rt_args=Dict[str, Union[str, List[str]]])`.

See [example_combined](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/examples/cli/plugins/example_combined) for an example.

### Execute Method

Contributors who are not overloading `BaseRT.benchmark()` must overload `BaseRT._execute()`. By default, `BaseRT` will automatically call `self._execute()` during `BaseRT.benchmark()`, which implements the specific benchmarking methodology for that device/runtime combination. For example, `tensorrt/runtime.py::_execute_()` implements benchmarking on Nvidia GPU devices with the TensorRT runtime.

Implementation of the execute method is optional, however if you do not implement the execute method you will have to overload `BaseRT.benchmark()` with your own functionality as in `TorchRT`.

`_execute()` must implement the following arguments (note that it is not required to make use of all of them):
- `output_dir`: path where the benchmarking artifacts (ONNX files, inputs, outputs, performance data, etc.) are located.
- `onnx_file`: path where the ONNX file for the model is located.
- `outputs_file`: path where the benchmarking outputs will be located.
- `iterations`: number of execution iterations of the model to capture the throughput and mean latency.

Additionally, `self._execute()` can access any custom runtime argument that has been added to `self` by the runtime class.

## Contributing to the overall framework
If you wish to contribute to any other part of the repository such as examples or reporting, please open an [issue](#issues) with the following details.

1. **Title:** A concise, descriptive title that summarizes the contribution.
1. **Tags/Labels:** Add any relevant tags or labels such as 'enhancement', 'good first issue', or 'help wanted'
1. **Proposal:** Detailed description of what you propose to contribute. For new examples, describe what they will demonstrate, the technology or tools they'll use, and how they'll be structured.

## Issues

Please file any bugs or feature requests you have as an [Issue](https://github.com/aig-bench/onnxmodelzoo/issues) and we will take a look.

## Pull Requests

Contribute code by creating a pull request (PR). Your PR will be reviewed by one of the [repo maintainers](https://github.com/aig-bench/onnxmodelzoo/blob/main/CODEOWNERS).

Please have a discussion with the team before making major changes. The best way to start such a discussion is to file an [Issue](https://github.com/aig-bench/onnxmodelzoo/issues) and seek a response from one of the [repo maintainers](https://github.com/aig-bench/onnxmodelzoo/blob/main/CODEOWNERS).

## Testing

Tests are defined in `tests/` and run automatically on each PR, as defined in our [testing action](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/.github/workflows/test.yml). This action performs both linting and unit testing and must succeed before code can be merged.

We don't have any fancy testing framework set up yet. If you want to run tests locally:
- Activate a `conda` environment that has `turnkey` (this package) installed.
- Run `conda install pylint` if you haven't already (other pylint installs will give you a lot of import warnings).
- Run `pylint src --rcfile .pylintrc` from the repo root.
- Run `python *.py` for each test script in `test/`.

## Versioning

We use semantic versioning, as described in [versioning.md](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/versioning.md).
