# Welcome to ONNX TurnkeyML

[![Turnkey tests](https://github.com/aig-bench/onnxmodelzoo/actions/workflows/test_turnkey.yml/badge.svg)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/test "Check out our tests")
[![Build API tests](https://github.com/aig-bench/onnxmodelzoo/actions/workflows/test_build_api.yml/badge.svg)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/test "Check out our tests")
[![Turnkey GPU tests](https://github.com/aig-bench/onnxmodelzoo/actions/workflows/test_gpu_turnkey.yml/badge.svg)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/test "Check out our tests")
[![OS - Linux](https://img.shields.io/badge/OS-Linux-blue?logo=linux&logoColor=white)](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/install.md "Check out our instructions")ADD WINDOWS
[![Made with Python](https://img.shields.io/badge/Python-3.8,3.10-blue?logo=python&logoColor=white)](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/install.md "Check out our instructions")


We are on a mission to understand and use as many models as possible while leveraging the right toolchain and AI hardware for the job in every scenario. 

Evaluating a deep learning model with a familiar toolchain and hardware accelerator is pretty straightforward. Scaling these evaluations to get apples-to-applies insights for thousands of models, permuted across dozens of toolchain and hardware targets, is not straightforward. 

TurnkeyML is a tools *framework* that integrates models, toolchains, and hardware backends to make evaluation and actuation of this landscape as simple as turning a key.

## Get started

For most users its as simple as:

```
conda activate ENV
(ENV) pip install turnkeyml
export models = $(turnkey models location --quiet)
turnkey $models/selftest/linear.py
```

The [installation guide](ADDLINK), [tutorials](ADDLINK), and [user guide](ADDLINK) have everything you need to know.

## Use Cases

TurnkeyML is designed to support the following use cases. Of course, it is also quite flexible, so we are sure you will come up with some use cases of your own too.

| Use Case               | Description | Example         |
|------------------------|-------------|-----------------|
| ONNX Model Zoo         | Export thousands of ONNX files across different opsets and data types. This is how we generated the contents of the new [ONNX Model Zoo](ADDLINK). | [Link](ADDLINK) |
| Model insights         | Analyze a model to learn its parameter count, input shapes, which ONNX ops it uses, etc. | [Link](ADDLINK) |
| Functional coverage    | Measure the functional coverage of toolchain/hardware combinations over a large corpus of models (e.g., how many models are supported by a novel compiler?). | [Link](ADDLINK) |
| Stress testing         | Run millions of inferences across thousands of models and log all the results to find the bugs in a HW/SW stack. | [Link](ADDLINK) |
| Performance validation | Measure latency and throughput in hardware across devices and runtimes to understand product-market fit. | [Link](ADDLINK) |


## Demo

Let's say you have some python script that includes a PyTorch model. Maybe you downloaded the model from Huggingface, grabbed it from our corpus, or wrote it yourself. Doesn't matter, just call `turnkey` and get to work.   

The `turnkey` CLI will analyze your script, find the model(s), run an ONNX toolchain on the model, and execute the resulting ONNX file in hardware:

> turnkey bert.py

```
Models discovered during profiling:

bert.py:
        model (executed 1x)
                Model Type:     Pytorch (torch.nn.Module)
                Class:          BertModel (<class 'transformers.models.bert.modeling_bert.BertModel'>)
                Location:       /home/jfowers/onnxmodelzoo/toolchain/models/transformers/bert.py, line 23
                Parameters:     109,482,240 (417.64 MB)
                Input Shape:    'attention_mask': (1, 128), 'input_ids': (1, 128)
                Hash:           bf722986
                Build dir:      /home/jfowers/.cache/turnkey/bert_bf722986
                Status:         Successfully benchmarked on AMD Ryzen 9 7940HS w/ Radeon 780M Graphics (ort v1.15.1) 
                                Mean Latency:   44.168  milliseconds (ms)
                                Throughput:     22.6    inferences per second (IPS)
```

Let's say you want an float16 ONNX file of the same model: incorporate the `oml-float16` (ONNX ML Tools fp16 converter tool) into the build sequence, and the `Build dir` will contain the ONNX file you seek:

> turnkey build bert.py --sequence export optimize-fp16

```
bert.py:
        model (executed 1x)
                ...
                Build dir:      /home/jfowers/.cache/turnkey/bert_bf722986
                Status:         Model successfully built!
```

Now you want to see the float16 model running on your Nvidia GPU with the Nvidia TensorRT runtime:

> turnkey bert.py --sequence export optimize-fp16 --device nvidia --runtime tensorrt

```
bert.py:
        model (executed 1x)
                ...
                Status:         Successfully benchmarked on NVIDIA GeForce RTX 4070 Laptop GPU (trt v23.09-py3) 
                                Mean Latency:   2.573   milliseconds (ms)
                                Throughput:     377.8   inferences per second (IPS)
```

And finally, mad with power, you want to see dozens of float16 Transformers running on your Nvidia GPU:

> turnkey REPO_ROOT/models/transformers/*.py --sequence export oml-float16 --device nvidia --runtime tensorrt

```
Models discovered during profiling:

albert.py:
        model (executed 1x)
                Class:          AlbertModel (<class 'transformers.models.albert.modeling_albert.AlbertModel'>)
                Parameters:     11,683,584 (44.57 MB)
                Status:         Successfully benchmarked on NVIDIA GeForce RTX 4070 Laptop GPU (trt v23.09-py3) 
                                Mean Latency:   1.143   milliseconds (ms)
                                Throughput:     828.3   inferences per second (IPS)

bart.py:
        model (executed 1x)
                Class:          BartModel (<class 'transformers.models.bart.modeling_bart.BartModel'>)
                Parameters:     139,420,416 (531.85 MB)
                Status:         Successfully benchmarked on NVIDIA GeForce RTX 4070 Laptop GPU (trt v23.09-py3) 
                                Mean Latency:   2.343   milliseconds (ms)
                                Throughput:     414.5   inferences per second (IPS)

bert.py:
        model (executed 1x)
                Class:          BertModel (<class 'transformers.models.bert.modeling_bert.BertModel'>)
                Parameters:     109,482,240 (417.64 MB)
                Status:         Successfully benchmarked on NVIDIA GeForce RTX 4070 Laptop GPU (trt v23.09-py3) 
                                Mean Latency:   2.565   milliseconds (ms)
                                Throughput:     378.0   inferences per second (IPS)

...
```

You're probably starting to get the idea :rocket:

There's a lot more features you can learn about in the [tutorials](ADDLINK) and [user guide](ADDLINK).

## What's Inside

The TurnkeyML framework has 4 core components:
- **Analysis tool**: Inspect Python scripts to find the PyTorch models within. Discover insights and pass the models to the other tools.
- **Build tool**: Export, optimize, quantize, and compile models. Any model-to-model transformation is fair game.
- **Runtime tool**: Execute models in hardware and measure key performance indicators.
- **Models corpus**: Hundreds of popular PyTorch models that are ready for use with `turnkey`.

All of this is seamlessly integrated together such that a command like `turnkey repo/models/corpus/script.py` gets you all of the functionality in one shot. Or you can access functionality piecemeal with commands and APIs like `turnkey analyze script.py` or `build_model(my_model_instance)`. The [tutorials](ADDLINK) show off the individual features.

You can read more about the code organization [here](ADDLINK).

## Extensibility

### Models

[![Transformers](https://img.shields.io/github/directory-file-count/aig-bench/onnxmodelzoo/toolchain/models/transformers?label=transformers)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/models/transformers "Transformer models")
[![Diffusers](https://img.shields.io/github/directory-file-count/aig-bench/onnxmodelzoo/toolchain/models/diffusers?label=diffusers)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/models/diffusers "Diffusion models")
[![popular_on_huggingface](https://img.shields.io/github/directory-file-count/aig-bench/onnxmodelzoo/toolchain/models/popular_on_huggingface?label=popular_on_huggingface)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/models/popular_on_huggingface "Popular Models on Huggingface")
[![torch_hub](https://img.shields.io/github/directory-file-count/aig-bench/onnxmodelzoo/toolchain/models/torch_hub?label=torch_hub)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/models/torch_hub "Models from Torch Hub")
[![torchvision](https://img.shields.io/github/directory-file-count/aig-bench/onnxmodelzoo/toolchain/models/torchvision?label=torchvision)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/models/torchvision "Models from Torch Vision")
[![timm](https://img.shields.io/github/directory-file-count/aig-bench/onnxmodelzoo/toolchain/models/timm?label=timm)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/models/timm "Pytorch Image Models")

This repository is home to a diverse corpus of hundreds of models. We are actively working on increasing the number of models on our model library. You can see the set of models in each category by clicking on the corresponding badge.

Evaluating a new model is as simple as taking a Python script that instantiates and invokes a PyTorch `torch.nn.module` and call `turnkey` on it. Read about model contributions [here](ADDLINK).

### Plugins

The build tool has built-in support for a variety of export and optimization tools (e.g., the PyTorch-to-ONNX exporter, ONNX ML Tools fp16 converter, etc.). Likewise, the runtime tool comes out-of-box with support for x86 and Nvidia devices, along with ONNX Runtime, TensorRT, torch-eager, and torch-compiled runtimes. 

If you need more, the TurnkeyML plugin API lets you extend the build and runtime tools with any functionality you like:

```
pip install -e my_custom_plugin
turnkey my_model.py --sequence my-custom-sequence --device my-custom-device --runtime my-custom-runtime --rt-args my-custom-args
```

All of the built-in sequences, runtimes, and devices are implemented against the plugin API. Check out this [example plugin](ADDLINK) and the [plugin API guide](ADDLINK).

## Contributing

We are actively seeking collaborators from across the industry. If you would like to contribute to this project, please check out our [contribution guide](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/contribute.md).

## Maintainers

This project is sponsored by the [ONNX Model Zoo](ADDLINK) special interest group (SIG). It is maintained by @danielholanda @jeremyfowers @ramkrishna in equal measure. You can reach us at [turnkeyml@???.com](ADDLINK) or by filing an [issue](ADDLINK).

## License

This project is licensed under the [Apache 2.0 License](ADDLINK).
