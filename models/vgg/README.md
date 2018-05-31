# VGG
## ONNX model
* [vgg16]() ; Size: 
* [vgg19]() ; Size:
* [vgg16_bn]() ; Size: 
* [vgg19_bn]() ; Size:
## MMS archive
* [vgg16]() ; Size: 
* [vgg19]() ; Size:
* [vgg16_bn]() ; Size: 
* [vgg19_bn]() ; Size:
## Dataset
Dataset used for train and validation: [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/). Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset.
## Example notebook
View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook
## Training notebook
Use the notebook [train_vgg](train_vgg.ipynb) to train the model. It contains the hyperparameters and network details used for training the above models
## Verify validation accuracy
Use the notebook [imagenet_verify](../imagenet_verify.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook
## References 
* **vgg16** and **vgg 19** are from the paper [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
* **vgg16_bn** and **vgg19_bn** are the same models as above but with batch normalization applied after each convolution layer
## Accuracy measures
## Tags
