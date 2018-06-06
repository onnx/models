# Mobilenet
<!-- add a description -->
<!-- why is there not the usage pattern for acquiring the model, serving the model, and running inference from the existing model zoo? Also what about the custom service and the inference input variable differences?:
https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md -->

## ONNX model
<!-- Consider using a table for these -->
* Version 2:
 Mobilenetv2-1.0 ; Size: 5MB
 Mobilenetv2-0.5 ; Size: 5MB
<!--links?-->

## MMS archive
* Version 2:   
 Mobilenetv2-1.0 ; Size: 5MB  
 Mobilenetv2-0.5 ; Size: 5MB
 <!--links?-->

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

## References
**Mobilenet-v2**
Model from the paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

## Accuracy Measures

## Tags
