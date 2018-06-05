# VGG
## ONNX model
* [vgg16]() ; Size: 
* [vgg19]() ; Size:
* [vgg16_bn]() ; Size: 
* [vgg19_bn]() ; Size:
## MMS archive
[MXNet Model Server](https://github.com/awslabs/mxnet-model-server) (MMS) is a flexible and easy tool to serve deep learning models from ONNX or MXNet. To learn about ONNX model serving with MMS, head over to this [doc](https://github.com/awslabs/mxnet-model-server/blob/master/docs/export_from_onnx.md). 
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
## Validation accuracy
The accuracies obtained by the models on the validation set are shown in the table below: 

|Model        |Top-1 error (%)|Top-5 error (%)|
|-------------|:--------------|:--------------|
|vgg16        |     27.38     |      8.86     |
|vgg16_bn     |     27.29     |      8.79     |
|vgg19        |     26.28     |      8.42     |
|vgg19_bn     |     26.16     |      8.21     |

Use the notebook [imagenet_verify](../imagenet_verify.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.
## References 
* **vgg16** and **vgg 19** are from the paper [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
* **vgg16_bn** and **vgg19_bn** are the same models as above but with batch normalization applied after each convolution layer
## Tags
