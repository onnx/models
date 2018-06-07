# VGG

VGG presents the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. VGG networks have increased depth with very small (3 × 3) convolution filters, which showed a significant improvement on the prior-art configurations achieved by pushing the depth to 16–19 weight layers. The work secured the first and the second places in the localisation and classification tracks respectively in ImageNet Challenge 2014. The representations from VGG generalise well to other datasets, where they achieve state-of-the-art results. 

## Model

 |Model        |ONNX Model  | Model archives|
|-------------|:--------------|:--------------|
|VGG 16|    [527.8 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.onnx)    |  [527.9 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.model)     |
|VGG 16_bn|    [527.9 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16-bn/vgg16-bn.onnx)    |  [527.9 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16-bn/vgg16-bn.model)     |
|VGG 19|    [548.1 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19/vgg19.onnx)    |  [548.1 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19/vgg19.model)     |
|VGG 19_bn|    [548.1 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19-bn/vgg19-bn.onnx)    |  [548.2 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19-bn/vgg19-bn.model)     |

## Dataset
Dataset used for train and validation: [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/). 

Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset.
## Example notebook
View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook
## Training notebook
Use the notebook [train_vgg](train_vgg.ipynb) to train the model. It contains the hyperparameters and network details used for training the above models
## Validation accuracy
The accuracies obtained by the models on the validation set are shown in the table below: 

|Model        |Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|
|VGG 16        |     72.62     |      91.14     |
|VGG 16_bn     |     72.71     |      91.21    |
|VGG 19        |     73.72     |      91.58     |
|VGG 19_bn     |     73.83    |      91.79     |

Use the notebook [imagenet_verify](../imagenet_verify.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.
## References 
* **VGG 16** and **vVGG 19** are from the paper [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
* **VGG 16_bn** and **VGG 19_bn** are the same models as above but with batch normalization applied after each convolution layer
## Keywords
