# MobileNet
<!-- add a description -->

## Model
* Version 2:

 |Model        |ONNX Model  | Model archives|
|-------------|:--------------|:--------------|
|MobileNetv2-1.0|    [13.6 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx)    |  [13.7 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.model)     |
|MobileNetv2-0.5|    [13.6 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-0.5/mobilenetv2-0.5.onnx)    |  [13.7 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-0.5/mobilenetv2-0.5.model)     |



## Dataset

Dataset used for training and validation: [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/).
<!-- this is not a link to the dataset -->

Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset.
<!-- imagenet prep does not exist -->

## Example Notebook
View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook.

## Training Notebook
View the notebook that can be used for training [here](train_mobilenet.ipynb). The notebook contains details for the parameters and the network for each of the above variants of ResNet.

## Verify Validation Accuracy
Use the notebook [imagenet_verify](../imagenet_verify.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook

## Validation accuracy
The accuracies obtained by the models on the validation set are shown in the table below: 
* Version 1:

 |Model        |Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|
|MobileNetv2-1.0|     70.94    |     89.99           |
|MobileNetv2-0.5|              |             |
=
## References
**MobileNet-v2**
Model from the paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)


## Keywords
