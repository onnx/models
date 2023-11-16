# TurnkeyML Code Structure

# Repo Organization

The TurnkeyML source code has a few major top-level directories:
- `docs`: documentation for the entire project.
- `examples`: example scripts for use with the TurnkeyML tools.
  - `examples/api`: examples scripts that invoke the benchmarking API to get the performance of models.
  - `examples/cli`: tutorial series starting in `examples/cli/readme.md` to help learn the `turnkey` CLI.
    - `examples/cli/scripts`: example scripts that can be fed as input into the `turnkey` CLI. These scripts each have a docstring that recommends one or more `turnkey` CLI commands to try out.
- `models`: the corpora of models that makes up the TurnkeyML models (see [the models readme](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/models/readme.md)).
  - Each subdirectory under `models` represents a corpus of models pulled from somewhere on the internet. For example, `models/torch_hub` is a corpus of models from [Torch Hub](https://github.com/pytorch/hub).
- `src/turnkey`: source code for the TurnkeyML tools (see [Benchmarking Tools](#benchmarking-tools) for a description of how the code is used).
  - `src/turnkeyml/analysis`: functions for profiling a model script, discovering model instances, and invoking `benchmark_model()` on those instances.
  - `src/turnkeyml/api`: implements the benchmarking APIs.
  - `src/turnkeyml/cli`: implements the `turnkey` CLI.
  - `src/turnkeyml/common`: functions common to the other modules.
  - `src/turnkeyml/version.py`: defines the package version number.
- `src/turnkeyml/build`: source code for the build API (see [Model Build Tool](#model-build-tool))
- `test`: tests for the TurnkeyML tools.
  - `test/analysis.py`: tests focusing on the analysis of model scripts.
  - `test/cli.py`: tests focusing on top-level CLI features.

# Benchmarking Tools

TurnkeyML provides two main tools, the `turnkey` CLI and benchmarking APIs. Instructions for how to use these tools are documented in the [Tools User Guide](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md), while this section is about how the source code is invoked to implement the tools. All of the code below is located under `src/turnkeyml/`.

1. The `turnkey` CLI is the comprehensive frontend that wraps all the other code. It is implemented in [cli/cli.py](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/src/turnkeyml/cli/cli.py).
1. The default command for `turnkey` CLI runs the `benchmark_files()` API, which is implemented in [files_api.py](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/src/turnkeyml/files_api.py).
    - Other CLI commands are also implemented in `cli/`, for example the `report` command is implemented in `cli/report.py`.
1. The `benchmark_files()` API takes in a set of scripts, each of which should invoke at least one model instance, to evaluate and passes each into the `evaluate_script()` function for analysis, which is implemented in [analyze/script.py](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/src/turnkeyml/analyze/script.py).
1. `evaluate_script()` uses a profiler to discover the model instances in the script, and passes each into the `benchmark_model()` API, which is defined in [model_api.py](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/src/turnkeyml/model_api.py).
1. The `benchmark_model()` API prepares the model for benchmarking (e.g., exporting and optimizing an ONNX file), which creates an instance of a `*Model` class, where `*` can be CPU, GPU, etc. The `*Model` classes are defined in [run/](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/src/turnkeyml/run/).
1. The `*Model` classes provide a `.benchmark()` method that benchmarks the model on the device and returns an instance of the `MeasuredPerformance` class, which includes the performance statistics acquired during benchmarking.
1. `benchmark_model()` and the `*Model` classes are built using [`build_model()`](#model-build-tool)

# Model Build Tool

The `build` module implements the `build_model()` API, which is an automated toolchain for building PyTorch/Keras/Hummingbird models into optimized ONNX files.

The `build` codebase is housed in the repo's `src/turnkeyml/build` directory. The API itself is in `src/turnkeyml/build_api.py`.

The mission of `build_model()` is to get your model ready to to use in the ONNX ecosystem with only 1 function call on 1 line of code. This involves taking all the complexity of working with multiple ML frameworks, the ONNX ecosystem, and other tools and abstracting all of it away from the user. As such, there is a fair amount of internal complexity, but we have done our best to organize in a way that is reasonable to understand, maintain, and most importantly, extend.

The first thing you need to know about the code is that it is all structured around a rocket ship pun. The `build_model()` function is built from a `Sequence` of `Stages` that must undergo `ignition` and then `launch()`.

The second thing you need to know is that although `build_model()` must be magically simple in the average case, it is also designed to be completely customizable when needed. Custom `Sequences` and `Stages` empower developers to add support for virtually any input format or model transformation.

## `build_model()` Function

We wanted to keep `build_model()`'s definition clean and easy to understand, so it is made up of a series of function calls to a sub-module called `ignition`.

These `ignition` calls start by making sure the call to `build_model()` is legit, by checking the environment, the arguments passed to `build_model()`, and so on.

The first impactful choices in `build_model()` take place during `model intake`, which looks at the `model`, `inputs`, and other arguments passed to `build_model()` to determine what kind of model this is, and what to do with it. A key result of `model intake` is a `Sequence`, which is the series of build `Stages` that `build_model()` will use to build the model. `Sequences` and `Stages` get their own section below.

Next, `ignition` looks into the `build cache` to determine whether this model needs to be built, or whether it has already been built and we can just reload it. The `build cache` is a location on disk that stores the output files and state from every call to `build_model()`, and this cache check is one of the most involved parts of the `build_model()` codebase.

If `build_model()` need to perform a build, the `sequence` from `model intake` comes into play. You can read more about sequences below, but at a high level:
* `build_model()` displays a `monitor` to help users keep track of progress through the sequence
* the sequence is "launched", meaning a series of `Stages` will each execute (`fire()`) to build the model

Finally, if the sequence is successful, build_model() will display a success message and then return a `Model` instance. You can read about `Model` below. On failure, build_model() will display an error message that should be as helpful and actionable as possible.

## Sequence and Stage Classes

All of the logic for actually building models is contained in `Stage` classes. Generally, each `Stage` is a model-to-model transformation. For example, the `ExportPytorchModel` `Stage` transforms a PyTorch model instance into an ONNX model file.

`Stages` are designed to be composable, for example, there are already a few ONNX-to-ONNX `Stages` defined in `src/turnkeyml/build/export.py` that could theoretically be composed in any order.

The `justbuildit` module also provides the `Sequence` class, which facilitates running a series of `Stages`. For example, in the first version of build_model(), a PyTorch model instance can be built into a `BaseModel` using a `Sequence` of 4 `Stage`s.

`Sequence`s can also be nested inside of other `Sequence`s. For example, we can define a PyTorch-to-ONNX `Sequence`, and then nest that inside of another `Sequence`, that, for example, could map a PyTorch model all the way into an optimized ONNX model.

Every `Stage` in build_model() is defined by inheriting the `Stage` base class. Each `Stage` must provide a unique name and a message to be displayed on the `monitor`, along with an overload of the `fire()` method. This `fire()` method is what the `Sequence` will call on your behalf when running the build.

`fire()` receives a single argument, an instance of `State` (which you can read more about below). `Stage`s can do many things, but generally speaking they should do some, or all, of:

* Take one or more artifact(s) from `state.intermediate_results` as input
* Produce one or more new artifact(s) and save them `state.intermediate_results`
* Raise an exception if the build should not continue
* Set `state.build_status` to 'successful' if the state of the build represents a working basis for a `BaseModel`
* Return an updated `state` instance

## `State` Class

The `State` class keeps track of the state of a `build_model()` build. `State` is also automatically saved to disk as a `state.yaml` file in the `build cache` whenever an attribute is modified. There are three key intentions behind this implementation and usage of `State`:

1. Easily pass critical information between `Stages` in a standardized way
1. Facilitate debugging by keeping the latest information and build decisions in one place on disk
1. Make it easy to collect and report statistics across a large number of builds