# Tools User Guide

The TurnkeyML package provides a CLI, `turnkey`, and Python API for benchmarking machine learning and deep learning models. This document reviews the functionality provided by the package. If you are looking for repo and code organization, you can find that [here](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/code.md).

For a hands-on learning approach, check out the [`turnkey` CLI tutorials](#https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md).

The tools currently support the following combinations of runtimes and devices:

<span id="devices-runtimes-table">

| Device Type | Device arg | Runtime                                                                               | Runtime arg                      | Specific Devices                              |
| ----------- | ---------- | ------------------------------------------------------------------------------------- | -------------------------------- | --------------------------------------------- |
| Nvidia GPU  | nvidia     | TensorRT<sup>†</sup>                                                                  | trt                              | Any Nvidia GPU supported by TensorRT          |
| x86 CPU     | x86        | ONNX Runtime<sup>‡</sup>, Pytorch Eager, Pytoch 2.x Compiled | ort, torch-eager, torch-compiled | Any Intel or AMD CPU supported by the runtime |
</span>

<sup>†</sup> Requires TensorRT >= 8.5.2  
<sup>‡</sup> Requires ONNX Runtime >= 1.13.1  
<sup>*</sup> Requires Pytorch >= 2.0.0  


# Table of Contents

- [Just Benchmark It](#just-benchmark-it)
- [The turnkey() API](#the-turnkey-api)
- [Definitions](#definitions)
- [Devices and Runtimes](#devices-and-runtimes)
- [Additional Commands and Options](#additional-commands-and-options)
- [Plugins](#plugins)

# Just Benchmark It

The simplest way to get started with the tools is to use our `turnkey` command line interface (CLI), which can take any ONNX file or any python script that instantiates and calls PyTorch model(s) and benchmark them on any supported device and runtime.

On your command line:

```
pip install turnkey
turnkey your_script.py --device x86
```

Example Output:

```
> Performance of YourModel on device Intel® Xeon® Platinum 8380 is:
> latency: 0.033 ms
> throughput: 21784.8 ips
```

Where `your_script.py` is a Python script that instantiates and executes a PyTorch model named `YourModel`. The benchmarking results are also saved to a `build directory` in the `build cache` (see [Build](#build)).

The `turnkey` CLI performs the following steps:
1. [Analysis](#analysis): profile the Python script to identify the PyTorch models within
2. [Build](#build): call the `benchmark_files()` [API](#the-turnkey-api) to prepare each model for benchmarking
3. [Benchmark](#benchmark): call the `benchmark_model()` [API](#the-turnkey-api) on each model to gather performance statistics

_Note_: The benchmarking methodology is defined [here](#benchmark). If you are looking for more detailed instructions on how to install turnkey, you can find that [here](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/install.md).

> For a detailed example, see the [CLI Hello World tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md#hello-world).

> `turnkey` can also benchmark ONNX files with a command like `turnkey your_model.onnx`. See the [CLI ONNX tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md#onnx-benchmarking) for details. However, the majority of this document focuses on the use case of passing .py scripts as input to `turnkey`.

# The TurnkeyML API

Most of the functionality provided by the `turnkey` CLI is also available in the the API:
- `turnkey.benchmark_files()` provides the same benchmarking functionality as the `turnkey` CLI: it takes a list of files and target device, and returns performance results.
- `turnkey.benchmark_model()` provides a subset of this functionality: it takes a model and its inputs, and returns performance results.
  - The main difference is that `benchmark_model()` does not include the [Analysis](#analysis) feature, and `benchmark_files()` does.
- `turnkey.build_model(model, inputs)` is used to programmatically [build](#build) a model instance through a sequence of model-to-model transformations (e.g., starting with an fp32 PyTorch model and ending with an fp16 ONNX model). 

Generally speaking, the `turnkey` CLI is a command line interface for the `benchmark_files()` API, which internally calls `benchmark_model()`, which in turn calls `build_model()`. You can read more about this code organization [here](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/code.md).

For an example of `benchmark_model()`, the following script:

```python
from turnkeyml import benchmark_model

model = YourModel() # Instantiate a torch.nn.module
results = model(**inputs)
perf = benchmark_model(model, inputs)
```

Will print an output like this:

```
> Performance of YourModel on device Intel® Xeon® Platinum 8380 is:
> latency: 0.033 ms
> throughput: 21784.8 ips
```

`benchmark_model()` returns a `MeasuredPerformance` object that includes members:
 - `latency_units`: unit of time used for measuring latency, which is set to `milliseconds (ms)`.
 - `mean_latency`: average benchmarking latency, measured in `latency_units`.
 - `throughput_units`: unit used for measuring throughput, which is set to `inferences per second (IPS)`.
 - `throughput`: average benchmarking throughput, measured in `throughput_units`.

 _Note_: The benchmarking methodology is defined [here](#benchmark).

 For an example of `build_model()`, the following script:

 ```python
 from turnkeyml import build_model

 model = YourModel() # Instantiate a torch.nn.module
 results = model(**inputs)
 build_model(model, inputs, sequence="onnx-fp32") # Export the model as an fp32 ONNX file
 ```

# Definitions

This package uses the following definitions throughout.

## Model

A **model** is a PyTorch (torch.nn.Module) instance that has been instantiated in a Python script, or a `.onnx` file.

- Examples: BERT-Base, ResNet-50, etc.

## Device

A **device** is a piece of hardware capable of running a model.

- Examples: Nvidia A100 40GB, Intel Xeon Platinum 8380

## Runtime

A **runtime** is a piece of software that executes a model on a device.

- Different runtimes can produce different performance results on the same device because:
  - Runtimes often optimize the model prior to execution.
  - The runtime is responsible for orchestrating data movement, device invocation, etc.
- Examples: ONNX Runtime, TensorRT, PyTorch Eager Execution, etc.

## Analysis

**Analysis** is the process by which `benchmark_files()` inspects a Python script or ONNX file and identifies the models within.

`benchmark_files()` performs analysis by running and profiling your file(s). When a model object (see [Model](#model) is encountered, it is inspected to gather statistics (such as the number of parameters in the model) and/or pass it to the `benchmark_model()` API for benchmarking.

> _Note_: the `turnkey` CLI and `benchmark_files()` API both run your entire python script(s) whenever python script(s) are passed as input files. Please ensure that these scripts are safe to run, especially if you got them from the internet.

> See the [Multiple Models per Script tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md#multiple-models-per-script) for a detailed example of how analysis can discover multiple models from a single script.

## Model Hashes

Each `model` in a `script` is identified by a unique `hash`. The `analysis` phase of `benchmark_files()` will display the `hash` for each model. The `build` phase will save exported models to into the `cache` according to the naming scheme `{script_name}_{hash}`.

For example:

```
turnkey example.py --analyze-only

> pytorch_model (executed 1x - 0.15s)
>        Model Type:     Pytorch (torch.nn.Module)
>        Class:          SmallModel (<class 'linear_auto.SmallModel'>)
>        Location:       linear_auto.py, line 19
>        Parameters:     55 (<0.1 MB)
>        Hash:           479b1332
```

## Labels

Each `script` may have one or more labels which correspond to a set of key-value pairs that can be used as attributes of that given script. Labels must be in the first line of a `.py` file and are identified by the pragma `#labels`. Keys are separated from values by `::` and each label key may have one or more label values as shown in the example below:

For example:

```
#labels domain::nlp author::google task::question_answering,translation
```

Once a script has been benchmarked, all labels that correspond to that script will also be stored as part of the cache folder.


## Build

**Build** is the process by which the `build_model()` API consumes a [model](#model) and produces ONNX files and other artifacts needed for benchmarking.

We refer to this collection of artifacts as the `build directory` and store each build in the  `cache` for later use.

We leverage ONNX files because of their broad compatibility with model frameworks (PyTorch, Keras, etc.), software (ONNX Runtime, TensorRT, etc.), and devices (CPUs, GPUs, etc.). You can learn more about ONNX [here](https://onnx.ai/).

The `build_model()` API includes the following steps:
1. Take a `model` object and a corresponding set of `inputs`*.
1. Check the cache for a successful build we can load. If we get a cache hit, the build is done. If no build is found, or the build in the cache is stale, continue.
1. Pass the `model` and `inputs` to the ONNX exporter corresponding to the `model`'s framework (e.g., PyTorch models use `torch.onnx.export()`).
1. Perform additional optional build steps, for example using [ONNX Runtime](https://github.com/microsoft/onnxruntime) and [ONNX ML tools](https://github.com/onnx/onnxmltools) to optimize the model and convert it to float16, respectively.
1. Save the successful build to the cache for later use.

> *_Note_: Each `build` corresponds to a set of static input shapes. `inputs` are passed into the `build_model()` API to provide those shapes.

The *build cache* is a location on disk that holds all of the artifacts from your builds. The default cache location is `~/.cache/turnkey` (see [Cache Directory](#cache-directory) for more information).
- Each build gets its own directory, named according to the `build_name` (which is typically auto-selected), in the cache.
- A build is considered stale (will not be loaded by default) under the following conditions:
    - The model, inputs, or arguments to `build_model()` have changed since the last successful build.
        - Note: a common cause of builds becoming stale is when `torch` or `keras` assigns random values to parameters or inputs. You can prevent the random values from changing by using `torch.manual_seed(0)` or `tf.random.set_seed(0)`.
    - The major or minor version number of TurnkeyML has changed, indicating a breaking change to builds.
- The artifacts produced in each build include:
    - Build state is stored in `*_state.yaml`, where `*` is the build's name.
    - Log files produced by each stage of the build (`log_*.txt`, where `*` is the name of the stage).
    - ONNX files (.onnx) produced by build stages,
    - Statistics about the build are stored in a `turnkey_stats.yaml` file.
    - etc.

`build_model()` takes a few arguments that are not found in the higher-level APIs. If you are using the CLI or higher-level APIs, those arguments are automatically generated on your behalf. You can read about these in [Build API Arguments](#build-api-arguments).

## Benchmark

*Benchmark* is the process by which the `benchmark_model()` API collects performance statistics about a [model](#model). Specifically, `benchmark_model()` takes a [build](#build) of a model and executes it on a target device using target runtime software (see [Devices and Runtimes](#devices-and-runtimes)).

By default, `benchmark_model()` will run the model 100 times to collect the following statistics:
1. Mean Latency, in milliseconds (ms): the average time it takes the runtime/device combination to execute the model/inputs combination once. This includes the time spent invoking the device and transferring the model's inputs and outputs between host memory and the device (when applicable).
1. Throughput, in inferences per second (IPS):  the number of times the model/inputs combination can be executed on the runtime/device combination per second.
    > - _Note_: `benchmark_model()` is not aware of whether `inputs` is a single input or a batch of inputs. If your `inputs` is actually a batch of inputs, you should multiply `benchmark_model()`'s reported IPS by the batch size.

# Devices and Runtimes

The tools can be used to benchmark a model across a variety of runtimes and devices, as long as the device is available and the device/runtime combination is supported.

## Available Devices

The tools support benchmarking on both locally installed devices (including x86 CPUs / NVIDIA GPUs).

If you are using a remote machine, it must:
- turned on
- be available via SSH
- include the target device
- have `miniconda`, `python>=3.8`, and `docker>=20.10` installed

When you call `turnkey` CLI or `benchmark_model()`, the following actions are performed on your behalf:
1. Perform a `build`, which exports all models from the script to ONNX and prepares for benchmarking.
1. Set up the benchmarking environment by loading a container and/or setting up a conda environment.
1. Run the benchmarks.

## Arguments

The following arguments are used to configure `turnkey` and the APIs to target a specific device and runtime:

### Devices

Specify a device type that will be used for benchmarking.

Usage:
- `turnkey benchmark INPUT_FILES --device TYPE`
  - Benchmark the model(s) in `INPUT_FILES` on a locally installed device of type `TYPE` (eg, a locally installed Nvidia device).

Valid values of `TYPE` include:
- `x86` (default): Intel and AMD x86 CPUs.
- `nvidia`: Nvidia GPUs.

> _Note_: The tools are flexible with respect to which specific devices can be used, as long as they meet the requirements in the [Devices and Runtimes table](#devices-runtimes-table).
>  - The `turnkey` CLI will simply use whatever device, of the given `TYPE`, is available on the machine.
>  - For example, if you specify `--device nvidia` on a machine with an Nvidia A100 40GB installed, then the tools will use that Nvidia A100 40GB device.

Also available as API arguments: 
- `benchmark_files(device=...)`
- `benchmark_model(device=...)`.

> For a detailed example, see the [CLI Nvidia tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md#nvidia-benchmarking).

### Runtimes

Indicates which software runtime should be used for the benchmark (e.g., ONNX Runtime vs. Torch eager execution for a CPU benchmark).

Usage:
- `turnkey benchmark INPUT_FILES --runtime SW`

Each device type has its own default runtime, as indicated below.
- Valid runtimes for `x86` device
  - `ort`: ONNX Runtime (default).
  - `torch-eager`: PyTorch eager execution.
  - `torch-compiled`: PyTorch 2.x-style compiled graph execution using TorchInductor.
- Valid runtimes for `nvidia` device
  - `trt`: Nvidia TensorRT (default).

This feature is also be available as an API argument: 
- `benchmark_files(runtime=[...])`
- `benchmark_model(runtime=...)`

> _Note_: Inputs to `torch-eager` and `torch-compiled` are not downcasted to FP16 by default. Downcast inputs before benchmarking for a fair comparison between runtimes.

# Additional Commands and Options

`turnkey` and the APIs provide a variety of additional commands and options for users.

The default usage of `turnkey` is to directly provide it with a python script, for example `turnkey example.py --device x86`. However, `turnkey` also supports the usage `turnkey COMMAND`, to accomplish some additional tasks.

_Note_: Some of these tasks have to do with the `cache`, which stores the `build directories` (see [Build](#build)).

The commands are:
- [`benchmark`](#benchmark-command) (default command): benchmark the model(s) in one or more files
- [`cache list`](#list-command): list the available builds in the cache
- [`cache print`](#print-command): print the state of a build from the cache
- [`cache delete`](#delete-command): delete one or more builds from the cache
- [`cache report`](#report-command): print a report in .csv format summarizing the results of all builds in a cache
- [`version`](#version-command): print the `turnkey` version number

You can see the options available for any command by running `turnkey COMMAND --help`.

## `benchmark` Command

The `benchmark` command supports the arguments from [Devices and Runtimes](#devices-and-runtimes), as well as:

### Input Files

Name of one or more script (.py), ONNX (.onnx) files to be benchmarked. You may also specify a (.txt) that lists (.py) and (.onnx) models separated by line breaks.

Examples: 
- `turnkey models/selftest/linear.py` 
- `turnkey models/selftest/linear.py models/selftest/twolayer.py` 
- `turnkey examples/cli/onnx/sample.onnx` 

Also available as an API argument:
- `benchmark_files(input_files=...)`

You may also use [Bash regular expressions](https://tldp.org/LDP/Bash-Beginners-Guide/html/sect_04_01.html) to locate the files you want to benchmark.

Examples:
- `turnkey *.py`
  - Benchmark all scripts which can be found at the current working directory.
- `turnkey models/*/*.py`
  - Benchmark the entire corpora of models.
- `turnkey *.onnx`
  - Benchmark all ONNX files which can be found at the current working directory.
- `turnkey selected_models.txt`
  - Benchmark all models listed inside the text file.

> See the [Benchmark Multiple Scripts tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/discovery.md#benchmark-multiple-scripts) for a detailed example.

You can also leverage model hashes (see [Model Hashes](#model-hashes)) to filter which models in a script will be acted on, in the following manner:
  - `turnkey example.py::hash_0` will only benchmark the model corresponding to `hash_0`.
  - You can also supply multiple hashes, for example `turnkey example.py::hash_0,hash_1` will benchmark the models corresponding to both `hash_0` and `hash_1`.

> _Note_: Using bash regular expressions and filtering model by hashes are mutually exclusive. To filter models by hashes, provide the full path of the Python script rather than a regular expression.

> See the [Filtering Model Hashes tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/discovery.md#filtering-model-hashes) for a detailed example.

Additionally, you can leverage labels (see [Labels](#labels)) to filter which models in a script will be acted on, in the following manner:
  - `turnkey *.py --labels test_group::a` will only benchmark the scripts labels with `test_group::a`.
  - You can also supply multiple labels, for example `turnkey *.py --labels test_group::a domain::nlp` only benchmark scripts that have both `test_group::a`, and `domain::nlp` labels.

> _Note_: Using bash regular expressions and filtering model by hashes are mutually exclusive. To filter models by hashes, provide the full path of the Python script rather than a regular expression.

> _Note_: ONNX file input currently supports only models of size less than 2 GB. ONNX files passed directly into `turnkey *.onnx` are benchmarked as-is without applying any additional build stages.

### Use Slurm

Execute the build(s) and benchmark(s) on Slurm instead of using local compute resources. Each input runs in its own Slurm job.

Usage:
- `turnkey benchmark INPUT_FILES --use-slurm`
  - Use Slurm to run turnkey on INPUT_FILES.
- `turnkey benchmark SEARCH_DIR/*.py --use-slurm`
  - Use Slurm to run turnkey on all scripts in the search directory. Each script is evaluated as its on Slurm job (ie, all scripts can be evaluated in parallel on a sufficiently large Slurm cluster).

Available as an API argument:
- `benchmark_files(use_slurm=True/False)` (default False)

> _Note_: Requires setting up Slurm as shown [here](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/install.md).

> _Note_: while `--use-slurm` is implemented, and we use it for our own purposes, it has some limitations and we do not recommend using it. Currently, `turnkey` has some Slurm to be configuration assumptions that we have not documented yet. Please contact the developers by [filing an issue](https://github.com/aig-bench/onnxmodelzoo/issues/new) if you need Slurm support for your project.

> _Note_: Slurm mode applies a timeout to each job, and will cancel the job move if the timeout is exceeded. See [Set the Timeout](#set-the-timeout)

### Process Isolation

Evaluate each `turnkey` input in its own isolated subprocess. This option allows the main process to continue on to the next input if the current input fails for any reason (e.g., a bug in the input script, the operating system running out of memory, incompatibility between a model and the selected benchmarking runtime, etc.). 

Usage:
- `turnkey benchmark INPUT_FILES --process-isolation --timeout TIMEOUT`

Also available as an API argument:
- `benchmark_files(process_isolation=True/False, timeout=...)` (`process_isolation`'s default is False, `timeout`'s default is 3600)

Process isolation mode applies a timeout to each subprocess. The default timeout is 3600 seconds (1 hour) and this default can be changed with the [timeout environment variable](#set-the-default-timeout). If the child process is still running when the timeout elapses, turnkey will terminate the child process and move on to the next input file. 

> _Note_: Process isolation mode is mutually exclusive with [Slurm mode](#use-slurm).

### Cache Directory

`-d CACHE_DIR, --cache-dir CACHE_DIR` build cache directory where the resulting build directories will be stored (defaults to ~/.cache/turnkey).

Also available as API arguments:
- `benchmark_files(cache_dir=...)`
- `benchmark_model(cache_dir=...)`
- `build_model(cache_dir=...)`

> See the [Cache Directory tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/cache.md#cache-directory) for a detailed example.

### Lean Cache

`--lean-cache` Delete all build artifacts except for log files after the build.

Also available as API arguments: 
- `benchmark_files(lean_cache=True/False, ...)` (default False)
- `benchmark_model(lean_cache=True/False, ...)` (default False)

> _Note_: useful for benchmarking many models, since the `build` artifacts from the models can take up a significant amount of hard drive space.

> See the [Lean Cache tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/cache.md#lean-cache) for a detailed example.

### Rebuild Policy

`--rebuild REBUILD` Sets a cache policy that decides whether to load or rebuild a cached build.

Takes one of the following values:
  - *Default*: `"if_needed"` will use a cached model if available, build one if it is not available,and rebuild any stale builds.
  - Set `"always"` to force `turnkey` to always rebuild your model, regardless of whether it is available in the cache or not.
  - Set `"never"` to make sure `turnkey` never rebuilds your model, even if there is a stale or failed build in the cache. `turnkey`will attempt to load any previously built model in the cache, however there is no guarantee it will be functional or correct.

Also available as API arguments: 
- `benchmark_files(rebuild=...)`
- `benchmark_model(rebuild=...)`
- `build_model(rebuild=...)`

### Sequence

Replaces the default build sequence in `build_model()`. In the CLI, this argument is a string, referring to a built-in build sequence. For API users, this argument is either a string or an instance of `Sequence` that defines a custom build sequence. 

Usage:
- `turnkey benchmark INPUT_FILES --sequence CHOICE`

Also available as API arguments:
- `benchmark_files(sequence=...)`
- `benchmark_model(sequence=...)`
- `build_model(sequence=...)`

### Set Script Arguments

Sets command line arguments for the input script. Useful for customizing the behavior of the input script, for example sweeping parameters such as batch size. Format these as a comma-delimited string.

Usage:
- `turnkey benchmark INPUT_FILES --script-args="--batch_size=8 --max_seq_len=128"`
  - This will evaluate the input script with the arguments `--batch_size=8` and `--max_seq_len=128` passed into the input script.

Also available as an API argument:
- `benchmark_files(script_args=...)`

> See the [Parameters documentation](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/models/readme.md#parameters) for a detailed example.

### Maximum Analysis Depth

Depth of sub-models to inspect within the script. Default value is 0, indicating to only analyze models at the top level of the script. Depth of 1 would indicate to analyze the first level of sub-models within the top-level models.

Usage:
- `turnkey benchmark INPUT_FILES --max-depth DEPTH`

Also available as an API argument:
- `benchmark_files(max_depth=...)` (default 0)

> _Note_: `--max-depth` values greater than 0 are only supported for PyTorch models.

> See the [Maximum Analysis Depth tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/discovery.md#maximum-analysis-depth) for a detailed example.

### ONNX Opset

ONNX opset to be used when creating ONNX files, for example when calling `torch.onnx.export`. 

Usage:
- `turnkey benchmark INPUT_FILES --onnx-opset 16`

Also available as API arguments:
- `benchmark_files(onnx_opset=...)`
- `benchmark_model(onnx_opset=...)`
- `build_model(onnx_opset=...)`

> _Note_: ONNX opset can also be set by an environment variable. The --onnx-opset argument takes precedence over the environment variable. See [TURNKEY_ONNX_OPSET](#set-the-onnx-opset).

### Iterations

Iterations takes an integer that specifies the number of times the model inference should be run during benchmarking. This helps in obtaining a more accurate measurement of the model's performance by averaging the results across multiple runs. Default set to 100 iterations per run. 

Usage:
- `turnkey benchmark INPUT_FILES --iterations 1000`

Also available as API arguments:
- `benchmark_files(iterations=...)`
- `benchmark_model(iterations=...)`

### Analyze Only

Instruct `turnkey` or `benchmark_model()` to only run the [Analysis](#analysis) phase of the `benchmark` command.

Usage:
- `turnkey benchmark INPUT_FILES --analyze-only`
  - This discovers models within the input script and prints information about them, but does not perform any build or benchmarking.

> _Note_: any build- or benchmark-specific options will be ignored, such as `--runtime`, `--device`, etc.

Also available as an API argument: 
- `benchmark_files(analyze_only=True/False)` (default False)

> See the [Analyze Only tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/discovery.md#analyze-only) for a detailed example.

### Build Only

Instruct `turnkey`, `benchmark_files()`, or `benchmark_model()` to only run the [Analysis](#analysis) and [Build](#build) phases of the `benchmark` command.

Usage:
- `turnkey benchmark INPUT_FILES --build-only`
  - This builds the models within the input script, however does not run any benchmark.

> _Note_: any benchmark-specific options will be ignored, such as `--runtime`.

Also available as API arguments:
- `benchmark_files(build_only=True/False)` (default False)
- `benchmark_model(build_only=True/False)` (default False)

> See the [Build Only tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/build.md#build-only) for a detailed example.

### Custom Runtime Arguments

Users can pass arbitrary arguments into a runtime, as long as the target runtime supports those arguments, by using the `--rt-args` argument.

None of the built-in runtimes support such arguments, however plugin contributors can use this interface to add arguments to their custom runtimes. See [plugins contribution guideline](https://github.com/aigdat/onnxmodelzoo/blob/main/toolchain/docs/contribute.md#contributing-a-plugin) for details.

Also available as API arguments:
- `benchmark_files(rt_args=Dict)` (default None)
- `benchmark_model(rt_args=Dict)` (default None)

## Cache Commands

The `cache` commands help you manage the `turnkey cache` and get information about the builds and benchmarks within it.

### `cache list` Command

`turnkey cache list` prints the names of all of the builds in a build cache. It presents the following options:

- `-d CACHE_DIR, --cache-dir CACHE_DIR` Search path for builds (defaults to ~/.cache/turnkey)

> _Note_: `cache list` is not available as an API.

> See the [Cache Commands tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/cache.md#cache-commands) for a detailed example.

### `cache stats` Command

`turnkey cache stats` prints out the selected the build's `state.yaml` file, which contains useful information about that build. The `state` command presents the following options:

- `build_name` Name of the specific build whose stats are to be printed, within the cache directory
- `-d CACHE_DIR, --cache-dir CACHE_DIR` Search path for builds (defaults to ~/.cache/turnkey)

> _Note_: `cache stats` is not available as an API.

> See the [Cache Commands tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/cache.md#cache-commands) for a detailed example.

### `cache delete` Command

`turnkey cache delete` deletes one or more builds from a build cache. It presents the following options:

- `build_name` Name of the specific build to be deleted, within the cache directory
- `-d CACHE_DIR, --cache-dir CACHE_DIR` Search path for builds (defaults to ~/.cache/turnkey)
- `--all` Delete all builds in the cache directory

> _Note_: `cache delete` is not available as an API.

> See the [Cache Commands tutorial](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/cache.md#cache-commands) for a detailed example.

### `cache clean` Command

`turnkey cache clean` removes the build artifacts from one or more builds from a build cache. It presents the following options:

- `build_name` Name of the specific build to be cleaned, within the cache directory
- `-d CACHE_DIR, --cache-dir CACHE_DIR` Search path for builds (defaults to ~/.cache/turnkey)
- `--all` Clean all builds in the cache directory

> _Note_: `cache clean` is not available as an API.

### `cache report` Command

`turnkey cache report` analyzes the state of all builds in a build cache and saves the result to a CSV file. It presents the following options:

- `-d CACHE_DIR, --cache-dir CACHE_DIR` Search path for builds (defaults to ~/.cache/turnkey)
- `-r REPORT_DIR, --report-dir REPORT_DIR` Path to folder where report will be saved (defaults to current working directory)

> _Note_: `cache report` is not available as an API.

### `cache location` Command

`turnkey cache location` prints out the location of the default cache directory.

> _Note_: Also available programmatically, with `turnkey.filesystem.DEFAULT_CACHE_DIR`

## Models Commands

The `models` commands help you work with the turnkey models provided in the package.

### `locate models` Command

`turnkey models location` prints out the location of the [models directory](https://github.com/aig-bench/onnxmodelzoo/tree/main/models) with over 1000 models. It presents the following options:

- `--quiet` Command output will only include the directory path

> _Note_: Also available programmatically, with `turnkey.filesystem.MODELS_DIR`

## `version` Command

`turnkey version` prints the version number of the installed `turnkey` package.

`version` does not have any options.

> _Note_: `version` is not available as an API.

## Environment Variables

There are some environment variables that can control the behavior of the tools.

### Overwrite the Cache Location

By default, the tools will use `~/.cache/turnkey` as the cache location. You can override this cache location with the `--cache-dir` and `cache_dir=` arguments for the CLI and APIs, respectively.

However, you may want to override cache location for future runs without setting those arguments every time. This can be accomplished with the `TURNKEY_CACHE_DIR` environment variable. For example:

```
export TURNKEY_CACHE_DIR=~/a_different_cache_dir
```

### Show Traceback

By default, `turnkey` and `benchmark_files()` will display the traceback for any exceptions caught during model build. However, you may sometimes want a cleaner output on your terminal. To accomplish this, set the `TURNKEY_TRACEBACK` environment variable to `False`, which will catch any exceptions during model build and benchmark and display a simple error message like `Status: Unknown turnkey error: {e}`. 

For example:

```
export TURNKEY_TRACEBACK=False
```

### Preserve Terminal Outputs

By default, `turnkey` and `benchmark_files()` will erase the contents of the terminal in order to present a clean status update for each script and model evaluated. 

However, you may want to see everything that is being printed to the terminal. You can accomplish this by setting the `TURNKEY_DEBUG` environment variable to `True`. For example:

```
export TURNKEY_DEBUG=True
```

### Set the ONNX Opset

By default, `turnkey`, `benchmark_files()`, and `benchmark_model()` will use the default ONNX opset defined in `turnkey.common.build.DEFAULT_ONNX_OPSET`. You can set a different default ONNX opset by setting the `TURNKEY_ONNX_OPSET` environment variable.

For example:

```
export TURNKEY_ONNX_OPSET=16
```

### Set the Default Timeout

`turnkey` and `benchmark_files()` apply a default timeout, `turnkey.cli.spawn.DEFAULT_TIMEOUT_SECONDS`, when evaluating each input file when in [Slurm](#use-slurm) or [process isolation](#process-isolation) modes. If the timeout is exceeded, evaluation of the current input file is terminated and the program moves on to the next input file.

This default timeout can be overridden by setting the `TURNKEY_TIMEOUT_SECONDS` environment variable. 

For example:

```
export TURNKEY_TIMEOUT_SECONDS=1800
```

would set the timeout to 1800 seconds (30 minutes).

### Disable the Build Status Monitor

`turnkey` and the APIs display a build status monitor that shows progress through the various build stages. This monitor can cause problems on some terminals, so you may want to disable it.

This build monitor can be disabled by setting the `TURNKEY_BUILD_MONITOR` environment variable to `"False"`. 

For example:

```
export TURNKEY_BUILD_MONITOR="False"
```

would set the timeout to 1800 seconds (30 minutes).

# Plugins

The tools support a variety of built-in build sequences, runtimes, and devices (see the [Devices and Runtimes table](#devices-runtimes-table)). However, you may be interested to add support for a different build sequence, runtime, or device of your choosing. This is supported through the plugin interface.

A turnkey plugin is a pip-installable package that implements support for building a model using a custom sequence and/or benchmarking a model on a [device](#device) with a [runtime](#runtime). These packages must adhere to a specific plugin template. 

For more details on implementing a plugin, please refer to the [plugins contribution guideline](https://github.com/aigdat/onnxmodelzoo/blob/main/toolchain/docs/contribute.md#contributing-a-plugin)

# Build API Arguments

This section documents the arguments to `build_model()` that are not shared in common with the higher level APIs.

## Set the Model Inputs

- Used by `build_model()` to determine the shape of input to build against.
- Dictates the maximum input size the model will support.
- Same exact format as your model inputs.
- Inputs provided here can be dummy inputs.
- Hint: At runtime, pad your inference inputs to this expected input size.

> Good: allows for an input length of up to 128

`inputs = tokenizer("I like dogs", padding="max_length", max_length=128)`

> Bad: allows for inputs only the size of "I like dogs" and smaller

`inputs = tokenizer("I like dogs")`

### Example:

```
build_model(my_model, inputs)
```

See
 - `examples/build_api/hello_pytorch_world.py`
 - `examples/build_api/hello_keras_world.py`

---

## Change the Build Name

By default, `build_model()` will use the name of your script as the name for your build in the `build cache`. For example, if your script is named `my_model.py`, the default build name will be `my_model`. The higher-level TurnkeyML APIs will also automatically assign a build name.

However, if you are using `build_model()` directly, you can also specify the name using the `build_name` argument. 

Additionally, if you want to build multiple models in the same script, you must set a unique `build_name` for each to avoid collisions. 

### Examples:

> Good: each build has its own entry in the `build cache`.

```
build_model(model_a, inputs_a, build_name="model_a")
build_model(model_b, inputs_b, build_name="model_b")
```

> Bad: the two builds will collide, and the behavior will depend on your [rebuild policy](#rebuild-policy)

- `rebuild="if_needed"` and `rebuild="always"` will replace the contents of `model_a` with `model_b` in the cache.
  - `rebuild="if_needed"` will also print a warning when this happens.
- `rebuild="never"` will load `model_a` from cache and use it to populate `model_b`, and print a warning.

```
build_model(model_a, inputs_a)
build_model(model_b, inputs_b)
```

See: `examples/build_api/build_name.py`

---

## Turn off the Progress Monitor

The `build_model()` API displays a monitor on the command line that updates the progress of of your build. By default, this monitor is on, however, it can be disabled using the `monitor` flag.

**monitor**
- *Default*: `build_model(monitor=True, ...)` displays a progress monitor on the command line.
- Set `build_model(monitor=False, ...)` to disable the command line monitor.

### Example:

```
build_model(model, inputs, monitor=False)
```

See: `examples/build_api/no_monitor.py`

---
