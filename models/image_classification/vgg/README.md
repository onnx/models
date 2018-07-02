# VGG

## Use cases
VGG models perform image classification - they take images as input and classify the major object in the image into a set of pre-defined classes. They are trained on ImageNet dataset which contains images from 1000 classes.
VGG models provide very high accuracies but at the cost of increased model sizes. They are ideal for cases when high accuracy of classification is essential and there are limited constraints on model sizes.

## Description
VGG presents the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. VGG networks have increased depth with very small (3 × 3) convolution filters, which showed a significant improvement on the prior-art configurations achieved by pushing the depth to 16–19 weight layers. The work secured the first and the second places in the localisation and classification tracks respectively in ImageNet Challenge 2014. The representations from VGG generalise well to other datasets, where they achieve state-of-the-art results.

## Model

The models below are variant of same network with different number of layers and use of batch normalisation. VGG 16 and VGG 19 have 16 and 19 convolutional layers respectively. VGG 16_bn and VGG 19_bn have the same architecture as their original counterparts but with batch normalization applied after each convolutional layer, which leads to better convergence and slightly better accuracies.

 |Model        |Download  | Download (with sample test data)|ONNX Version|Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|VGG 16|    [527.8 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.onnx)    | [490.0 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.tar.gz)| 1.2.1  | 72.62     |      91.14     |
|VGG 16_bn|    [527.9 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16-bn/vgg16-bn.onnx) |[490.2 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16-bn/vgg16-bn.tar.gz)   |  1.2.1  |   72.71     |      91.21    |
|VGG 19|    [548.1 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19/vgg19.onnx)    | [508.5 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19/vgg19.tar.gz)| 1.2.1   | 73.72     |      91.58     |
|VGG 19_bn|    [548.1 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19-bn/vgg19-bn.onnx) |[508.6 MB](https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19-bn/vgg19-bn.tar.gz)   |  1.2.1    | 73.83    |      91.79     |

## Inference
We used MXNet as framework with gluon APIs to perform inference. View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook. 

### Input 
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (N x 3 x H x W), where N is the batch size, and H and W are expected to be at least 224.
The inference was done using jpeg image.

### Preprocessing
The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. The transformation should preferrably happen at preprocessing. Check [imagenet_preprocess.py](../imagenet_preprocess.py) for code.

### Output
The model outputs image scores for each of the [1000 classes of ImageNet](../synset.txt).

### Postprocessing
The post-processing involves calculating the softmax probablility scores for each class and sorting them to report the most probable classes. Check [imagenet_postprocess.py](../imagenet_postprocess.py) for code.

To do quick inference with the model, check out [Model Server](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md/#vgg_header).

## Dataset
Dataset used for train and validation: [ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset.

## Validation accuracy
The accuracies obtained by the models on the validation set are mentioned above. The accuracies have been calculated on center cropped images with a maximum deviation of 0.4% (top-1 accuracy) from the paper.

<!--|Model        |Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|
|VGG 16        |     72.62     |      91.14     |
|VGG 16_bn     |     72.71     |      91.21    |
|VGG 19        |     73.72     |      91.58     |
|VGG 19_bn     |     73.83    |      91.79     |
-->


## Training
We used MXNet as framework with gluon APIs to perform training. View the [training notebook](train_vgg.ipynb) to understand details for parameters and network for each of the above variants of VGG.

## Validation
We used MXNet as framework with gluon APIs to perform validation. Use the notebook [imagenet_validation](../imagenet_validation.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.


## References 
* **VGG 16** and **VGG 19** are from the paper [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
* **VGG 16_bn** and **VGG 19_bn** are the same models as above but with batch normalization applied after each convolution layer

## Contributors
* [abhinavs95](https://github.com/abhinavs95) (Amazon AI)
* [ankkhedia](https://github.com/ankkhedia) (Amazon AI)

## Acknowledgments
[MXNet](http://mxnet.incubator.apache.org), [Gluon model zoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html), [GluonCV](https://gluon-cv.mxnet.io)

## Keywords
CNN, VGG, ONNX, ImageNet, Computer Vision

## License
Apache 2.0