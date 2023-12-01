<!--- SPDX-License-Identifier: Apache-2.0 -->


# ONNX Model Zoo


## Introduction

Welcome to the ONNX Model Zoo! The Open Neural Network Exchange (ONNX) is an open standard format created to represent machine learning models. Supported by a robust community of partners, ONNX defines a common set of operators and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

This repository is a curated collection of pre-trained, state-of-the-art models in the ONNX format. These models are sourced from prominent open-source repositories and have been contributed by a diverse group of community members. Our aim is to facilitate the spread and usage of machine learning models among a wider audience of developers, researchers, and enthusiasts.

To handle ONNX model files, which can be large, we use Git LFS (Large File Storage). To download any model from this collection, please navigate to the specific model's GitHub page and click the "Download" button located in the top right corner, or visit the [ONNX Model Zoo website](insert-link-here).

## Models

Our Model Zoo covers a variety of use cases including:

- Computer Vision
- Natural Language Processing (NLP)
- Generative AI
- Graph Machine Learning

These models are sourced from prominent open-source repositories such as [timm](https://github.com/huggingface/pytorch-image-models), [torchvision](https://github.com/pytorch/vision), [torch_hub](https://pytorch.org/hub/), and [transformers](https://github.com/huggingface/transformers), and exported into the ONNX format using the open-source [TurnkeyML toolchain](insert-link-here).


## Usage

There are multiple ways to access the ONNX Model Zoo:

### Webpage (Recommended)

The [Model Zoo webpage](link-here) provides a user-friendly interface to browse models based on use case, author, or by searching for specific model names. It also offers a direct download option for your convenience.

### Git Clone (Not Recommended)

Cloning the repository using git won't automatically download the ONNX models due to their size. To manage these files, first, install Git LFS by running:

```bash
pip install git-lfs
```

To download a specific model:

```bash
git lfs pull --include="[path to model].onnx" --exclude=""
```

To download all models:

```bash
git lfs pull --include="*" --exclude=""
```

### GitHub UI

Alternatively, you can download models directly from GitHub. Navigate to the model's page and click the "Download" button on the top right corner.

## Model Visualization

For a graphical representation of each model's architecture, we recommend using [Netron](https://github.com/lutzroeder/netron).

## Contributions

Contributions to the ONNX Model Zoo are welcome! Please check our [contribution guidelines](here) for more information on how you can contribute to the growth and improvement of this resource.

Thank you for your interest in the ONNX Model Zoo, and we look forward to your participation in our community!

# License

[Apache License v2.0](LICENSE)
