<!--- SPDX-License-Identifier: Apache-2.0 -->

# Contribution Guide for the ONNX Model Zoo

## Introduction

We're excited that you're interested in contributing to the ONNX Model Zoo! This guide is intended to help you understand how you can contribute to the project and what standards we expect contributors to follow.

## Ways to Contribute

There are many ways you can contribute to the ONNX Model Zoo:

- **Model Contributions**: You can contribute new models to the Model Zoo.

- **Documentation**: Improvements to documentation, whether it's enhancing existing documentation or creating new content, are always welcome.

- **Issues and Bugs**: Report issues or bugs you find when using the models or the repository itself.

- **Features and Enhancements**: Suggest new features or enhancements to existing models or the overall repository infrastructure.

## Contribution Process

1. **Check Existing Issues/PRs**: Before starting your work or submitting a contribution, please check the repository for existing issues or pull requests that might be related to your contribution to avoid duplication of efforts.

2. **Open an Issue**: If you are adding a new feature or model, start by opening a new issue to discuss with the community. For bugs, document how to reproduce the error.

3. **Fork the Repository**: Make a fork of this repository to your GitHub account.

4. **Make Your Changes**: Work on the changes on your forked repository. 
    
    For contributing a model refer to [Model contribution guidelines](#model-contribution-guidelines) and for contributing to the toolchain under `onnx\models\toolchain` plese refer to the contribution guidelines here. (todo: add link)

5. **Commit Messages**: Use clear and descriptive commit messages. This helps to understand the purpose of your changes and speed up the review process.

6. **Pull Request**: Once you've made your changes, submit a pull request (PR) to the main repository. Provide a detailed description of the changes and reference the corresponding issue number in the PR.

7. **Code Review**: Wait for the maintainers to review your PR. Be responsive to feedback and make necessary changes if requested.

8. **Merge**: Once your PR is approved by a maintainer, it will be merged into the repository.

## Model Contribution Guidelines

To contribute a model to the ONNX Model Zoo, we ask that you adhere to the following guidelines to ensure consistency and quality:

1. **Model Uniqueness**: Before submission, please search the Model Zoo to confirm that your model or an equivalent model is not already hosted.

1. **Model Source**: You should provide the source file for the model (.py format).

1. **Source File**: Place the source file under `onnx\models\toolchain\models` and place the onnx file under the appropriate task category in the repo root. All pretrained models should include valid weights.

The source file must include the following labels at the top of the file:
    
    - Author - Name of the organization or the creator of the model
    - Task - For example, Computer Vision, Natural Language Processing (NLP), Audio Processing, MultiModal, Generative AI ,Graph Machine Learning ...
    - License: Submissions should be under one of the following licenses: Apache-2.0, BSD-3, or MIT.
    
    For the correct format of these labels, please refer to the example here (todo: add link).

1. **Dependencies**: Add any additional requirements run your model to `onnx\models\toolchain\model\requirements.txt`(todo: insert link) file.

1. **Documentation**: While optional, we highly encourage you to provide a README file with:
    - A detailed description of the model.
    - Instructions for use, including setup and execution.

1. **Large Files**: If your model includes large weight files, manage them with Git LFS.

1. **Model Building**: Use TurnkeyML's `--build-only` (todo: add link) option to generate the ONNX model. This ensures that the model conforms to standards compatible with the Model Zoo's infrastructure.


By following these guidelines, you help us maintain a high standard for the models in the ONNX Model Zoo, making it a valuable resource for everyone. We appreciate your contributions and look forward to including your models in our collection.

## Questions?

If you have any questions or need further clarification about contributing, please don't hesitate to open an issue.

Thank you for contributing to the ONNX Model Zoo!
