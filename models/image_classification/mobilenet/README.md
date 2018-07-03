# MobileNet

## Use cases
Mobilenet models perform image classification - they take images as input and classify the major object in the image into a set of pre-defined classes. They are trained on ImageNet dataset which contains images from 1000 classes. MobileNet models are also very efficient in terms of speed and size and hence are ideal for embedded and mobile applications.

## Description
MobileNet improves the state-of-the-art performance of mobile models on multiple tasks and benchmarks as well as across a spectrum of different model sizes. MobileNet is based on an inverted residual structure where the shortcut connections are between the thin bottleneck layers. The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. Additionally,  it removes non-linearities in the narrow layers in order to maintain representational power. 

## Model
MobileNet reduces the dimensionality of a layer thus reducing the dimensionality of the operating space. The  trade off between computation and accuracy is exploited in Mobilenet via a width multiplier parameter approach which allows one to reduce the dimensionality of the activation space until the manifold of interest spans this entire space.
The below model is using multiplier value as 1.0.
* Version 2:

 |Model        |Download  |Download (with sample test data)| ONNX version|Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|MobileNet v2-1.0|    [13.6 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx)    |  [14.1 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz) |  1.2.1   | 70.94    |     89.99           |
<!--
|MobileNet v2-0.5|    [13.6 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-0.5/mobilenetv2-0.5.onnx)    |  [13.7 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-0.5/mobilenetv2-0.5.model)     |          |             |
-->

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

To do quick inference with the model, check out [Model Server](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md/#mobilenetv2-1.0_onnx).

## Dataset
Dataset used for train and validation: [ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset.


## Validation accuracy
The accuracies obtained by the model on the validation set are mentioned above. The accuracies have been calculated on center cropped images with a maximum deviation of 1% (top-1 accuracy) from the paper.

## Training
We used MXNet as framework with gluon APIs to perform training. View the [training notebook](train_mobilenet.ipynb) to understand details for parameters and network for each of the above variants of MobileNet.

## Validation
We used MXNet as framework with gluon APIs to perform validation. Use the notebook [imagenet_validation](../imagenet_validation.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.


## References
**MobileNet-v2**
Model from the paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

## Contributors
* [ankkhedia](https://github.com/ankkhedia) (Amazon AI)
* [abhinavs95](https://github.com/abhinavs95) (Amazon AI)

## Acknowledgments
[MXNet](http://mxnet.incubator.apache.org), [Gluon model zoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html), [GluonCV](https://gluon-cv.mxnet.io)

## Keyword
CNN, MobileNet, ONNX, ImageNet, Computer Vision

## License
Apache 2.0
