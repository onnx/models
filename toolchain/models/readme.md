# TurnkeyML Models

This directory contains the TurnkeyML models, which is a large collection of models that can be evaluated using the [`turnkey` CLI tool](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md).

## Table of Contents

- [Turnkey Models](#turnkey-models)
  - [Table of Contents](#table-of-contents)
  - [Benchmark Organization](#benchmark-organization)
  - [Running the Benchmark](#running-the-benchmark)
    - [Prerequisites](#prerequisites)
    - [Benchmarking Commands](#benchmarking-commands)
  - [Model Template](#model-template)
    - [Input Scripts](#input-scripts)
    - [Labels](#labels)
    - [Parameters](#parameters)
    - [Example Script](#example-script)

## Benchmark Organization

The TurnkeyML collection is made up of several corpora of models (_corpora_ is the plural of _corpus_... we had to look it up too). Each corpus is named after the online repository that the models were sourced from. Each corpus gets its own subdirectory in the `models` directory. 

The corpora are:
- `diffusers`: models from the [Huggingface `diffusers` library](https://huggingface.co/docs/diffusers/index), including the models that make up Stable Diffusion.
- `graph_convolutions`: Graph Neural Network (GNN) models from a variety of publications. See the docstring on each .py file for the source.
- `popular_on_huggingface`: hundreds of the most-downloaded models from the [Huggingface models repository](https://huggingface.co/models).
- `selftest`: a small corpus with small models that can be used for testing out the tools.
- `torch_hub`: a variety of models, including many image classification models, from the [Torch Hub repository](https://github.com/pytorch/hub).
- `torchvision`: image recognition models from the [`torchvision` library](https://pytorch.org/vision/stable/index.html).
  - _Note_: the `torchvision` library also includes many image classification models, but we excluded them to avoid overlap between our `torch_hub` and `torchvision` corpora.
- `transformers`: Transformer models from the [Huggingface `transformers` library](https://huggingface.co/docs/transformers/index).
- `timm`: A collection of SOTA computer vision models from the [`timm` library](https://timm.fast.ai/).

## Running the Benchmark

### Prerequisites

Before running the benchmark we suggest you:
1. Install the `turnkey` package by following the [install instructions](https://github.com/aig-bench/onnxmodelzoo/tree/main/docs/install.md).
1. Go through the [`turnkey` CLI tutorials](https://github.com/aig-bench/onnxmodelzoo/tree/main/examples/cli/readme.md).
1. Familiarize yourself with the [`turnkey` CLI tool](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/turnkey_user_guide.md) documentation.

You must also run the following command to install all of the models' dependencies into your Python environment.

`pip install -r models/requirements.txt`

### Benchmarking Commands

Once you have fulfilled the prerequisites, you can evaluate one model from the benchmark with a command like this:

```
cd OMZ_ROOT/toolchain/models # OMZ_ROOT is where you cloned onnxmodelzoo
turnkey selftest/linear.py
```

You can also run the entire all models in one shot with:
```
cd OMZ_ROOT/toolchain/models # OMZ_ROOT is where you cloned onnxmodelzoo
turnkey */*.py
```

_Note_: Benchmarking the entire corpora of models might take a very long time.

You can aggregate all of the benchmarking results from your `turnkey cache` into a CSV file with:

```
turnkey report
```

If you want to only report on a subset of models, we recommend saving the benchmarking results into a specific cache directory:

```
# Save benchmark results into a specific cache directory
turnkey models/selftest/*.py -d selftest_results

# Report the results from the `selftest_results` cache
turnkey report -d selftest_results
```

If you have multiple cache directories, you may also aggregate all information into a single report:

```
turnkey report -d x86_results_cache_dir nvidia_results_cache_dir
```

## Model Template

Each model follows a template to keep things consistent. This template is meant to be as simple as possible. The models themselves should also be instantiated and called in a completely vendor-neutral manner.

### Input Scripts

Each model in the is hosted in a Python script (.py file). This script must meet the following requirements:
1. Instantiate at least one model and invoke it against some inputs. The size and shape of the inputs will be used for benchmarking.
  - _Note_: `turnkey` supports multiple models per script, and will benchmark all models within the script.
1. Provide a docstring that provides information about where the model was sourced from.

Each script can optionally include a set of [labels](#labels) and [parameters](#parameters). See [Example Script](#example-script) for an example of a well-formed script that instantiates one model.

_Note_: All of the scripts in `models/` are also functional on their own, without the `turnkey` command. For example, if you run the command:

```
python models/transformers_pytorch/bert.py
```

this will run the PyTorch version of the Huggingface `transformers` BERT model on your local CPU device.

### Labels

The models use labels to help organize the results data. Labels must be in the first line of the Python file and start with `# labels: `

Each label must have the format `key::value1,value2,...`

Example:

```
# labels: author::google test_group::daily,monthly
```
     
Labels are saved in your cache directory and can later be retrieved using the function `turnkey.common.labels.load_from_cache()`, which receives the `cache_dir` and `build_name` as inputs and returns the labels as a dictionary. 

### Parameters

The tools support parameterizing models so that you can sweep over interesting properties such as batch size, sequence length, image size, etc. The `turnkey.parser.parse()` method parses a set of standardized parameters that we have defined and provides them to the model as it is instantiated.

For example, this code would retrieve user-defined `batch_size` and `max_seq_length` (maximum sequence length) parameters.

```
from turnkeyml.parser import parse
parse(["batch_size", "max_seq_length"])
```

You can pass parameters into a benchmarking run with the `--script-args` argument to `turnkey`. For example, the command:

```
turnkey models/transformers_pytorch/bert.py --script-args="--batch_size 8 --max_seq_length 128"
```

would set `batch_size=8` and `max_seq_length=128` for that benchmarking run.

You can also use these arguments outside of `turnkey`, for example the command:

```
python models/transformers_pytorch/bert.py --batch_size 8
```

would simply run BERT with a batch size of 8 in PyTorch.


The standardized set of arguments is:

- General args
    - "batch_size": Arg("batch_size", default=1, type=int),
        - Batch size for the input to the model that will be used for benchmarking
    - "max_seq_length": Arg("max_seq_length", default=128, type=int),
        - Maximum sequence length for the model's input; also the input sequence length that will be used for benchmarking
    - "max_audio_seq_length": Arg("max_audio_seq_length", default=25600, type=int),
        - Maximum sequence length for the model's audio input; also the input sequence length that will be used for benchmarking
    - "height": Arg("height", default=224, type=int),
        - Height of the input image that will be used for benchmarking
    - "num_channels": Arg("num_channels", default=3, type=int),
        - Number of channels in the input image that will be used for benchmarking
    - "width": Arg("width", default=224, type=int),
        - Width of the input image that will be used for benchmarking
    - "pretrained": Arg("pretrained", default=False, type=bool),
        - Indicates whether pretrained weights should be used on the model
- Args for Graph Neural Networks
    - "k": Arg("k", default=8, type=int),
    - "alpha": Arg("alpha", default=2.2, type=float),
    - "out_channels": Arg("out_channels", default=16, type=int),
    - "num_layers": Arg("num_layers", default=8, type=int),
    - "in_channels": Arg("in_channels", default=1433, type=int),- 

### Example Script

The following example, copied from `models/transformers/bert.py` is a good example of a well-formed input script. You can see that it has the following properties:

1. Labels in the top line of the file.
1. Docstring indicating where the model was sourced from.
1. `turnkey.parser.parse()` is used to parameterize the model.
1. The model is instantiated and invoked against a set of inputs.

```
# labels: test_group::turnkey name::bert author::huggingface_pytorch
"""
https://huggingface.co/docs/transformers/v4.26.1/en/model_doc/bert#overview
"""
from turnkeyml.parser import parse
import transformers
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.BertConfig()
model = transformers.BertModel(config)
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
```
