# Mobilenet
## ONNX model
* Version 2:   
 Mobilenetv2-1.0 ; Size: 5MB  
 Mobilenetv2-0.5 ; Size: 5MB
 

## MMS archive
* Version 2:   
 Mobilenetv2-1.0 ; Size: 5MB  
 Mobilenetv2-0.5 ; Size: 5MB
 
## Dataset
Dataset used for train and validation: [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/). Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset. 
## Example notebook
View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook
## Training notebook
View the notebook that can be used for training [here](train_mobilenet.ipynb). The notebook contains details for 
parameters and network for each of the above variants of Resnet.
## Verify validation accuracy
Use the notebook [imagenet_verify](../imagenet_verify.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook
## References
**Mobilenet-v2**  
Model from the paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)  
 
 
 
## Accuracy measures
## Tags
