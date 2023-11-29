# The TurnkeyML Project

[![Turnkey tests](https://github.com/aig-bench/onnxmodelzoo/actions/workflows/test_turnkey.yml/badge.svg)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/test "Check out our tests")
[![Build API tests](https://github.com/aig-bench/onnxmodelzoo/actions/workflows/test_build_api.yml/badge.svg)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/test "Check out our tests")
[![Turnkey GPU tests](https://github.com/aig-bench/onnxmodelzoo/actions/workflows/test_gpu_turnkey.yml/badge.svg)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/test "Check out our tests")
[![OS - Linux](https://img.shields.io/badge/OS-Linux-blue?logo=linux&logoColor=white)](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/install.md "Check out our instructions")
[![Made with Python](https://img.shields.io/badge/Python-3.8,3.10-blue?logo=python&logoColor=white)](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/install.md "Check out our instructions")


TurnkeyML examines the capability of vendors to provide turnkey solutions to a corpus of hundreds of off-the-shelf models. All of the model scripts and benchmarking code are published as open source software.


## Benchmarking Tool

Our _turnkey_ CLI allows you to benchmark Pytorch models without changing a single line of code. The demo below shows BERT-Base being benchmarked on both Nvidia A100 and Intel Xeon. For more information, check out our [Tutorials](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md) and [Tools User Guide](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/tools_user_guide.md).

You can reproduce a nice demo by trying out the [Just Benchmark BERT](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/examples/cli/readme.md#just-benchmark-bert) tutorial.

## 1000+ Models

[![Transformers](https://img.shields.io/github/directory-file-count/aig-bench/onnxmodelzoo/toolchain/models/transformers?label=transformers)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/models/transformers "Transformer models")
[![Diffusers](https://img.shields.io/github/directory-file-count/aig-bench/onnxmodelzoo/toolchain/models/diffusers?label=diffusers)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/models/diffusers "Diffusion models")
[![popular_on_huggingface](https://img.shields.io/github/directory-file-count/aig-bench/onnxmodelzoo/toolchain/models/popular_on_huggingface?label=popular_on_huggingface)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/models/popular_on_huggingface "Popular Models on Huggingface")
[![torch_hub](https://img.shields.io/github/directory-file-count/aig-bench/onnxmodelzoo/toolchain/models/torch_hub?label=torch_hub)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/models/torch_hub "Models from Torch Hub")
[![torchvision](https://img.shields.io/github/directory-file-count/aig-bench/onnxmodelzoo/toolchain/models/torchvision?label=torchvision)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/models/torchvision "Models from Torch Vision")
[![timm](https://img.shields.io/github/directory-file-count/aig-bench/onnxmodelzoo/toolchain/models/timm?label=timm)](https://github.com/aig-bench/onnxmodelzoo/tree/main/toolchain/models/timm "Pytorch Image Models")

This repository is home to a diverse corpus of hundreds of models. We are actively working on increasing the number of models on our model library. You can see the set of models in each category by clicking on the corresponding badge.

## Installation

Please refer to our [turnkeyml installation guide](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/install.md) to get instructions on how to install the turnkeyml package.

## Contributing

We are actively seeking collaborators from across the industry. If you would like to contribute to this project, please check out our [contribution guide](https://github.com/aig-bench/onnxmodelzoo/blob/main/toolchain/docs/contribute.md).

## License

This is a closed source project. The source code and artifacts are AMD internal only. Do not distribute.
