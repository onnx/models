<p align="center">
<img src="images/ONNX Model Zoo Graphics.png" width="90%"/>
</p>

# Open Neural Network eXchange (ONNX) Model Zoo

The ONNX Model Zoo is a collection of pre-trained models for state of the art works in deep learning. Models are available in the ONNX protobuf format. Where available and supported, the models are also available in the MXNet Model Server (MMS) archive format. Accompanying each model are [Jupyter](http://jupyter.org) notebooks for model training and running inference with the trained model. The notebooks are written in Python using [MXNet](http://mxnet.incubator.apache.org) as a backend and include links to the training dataset as well as references to the original paper that describes the model architecture.

## Model Serving with MMS
Many of the models in this model zoo can be served with [MXNet Model Server](https://github.com/awslabs/mxnet-model-server) (MMS). MMS is a flexible and easy tool to serve deep learning models by providing a REST API with an inference end point. Supported ONNX models such as those converted from Chainer, CNTK, MXNet, and PyTorch can be served with MMS. To learn about ONNX model serving with MMS, refer to the [MMS ONNX documentation](https://github.com/awslabs/mxnet-model-server/blob/master/docs/export_from_onnx.md). 

## Model Visualization
You can see visualizations of each model's network architecture by using [Netron](https://lutzroeder.github.io/Netron).

## Contributions
Do you want to contribute a model? Check out the list of [backlog models](backlogs.md) <!-- should definitely pick a different name for this-->to get started. Also refer to the [contribution guidelines](contribute.md) before submitting a model.

## Models
### Image Classification
The model takes images as input and output the probability of it belonging to a set of pre-defined classes.

| | | |
|-|-|-|
|<b>[Squeezenet](models/squeezenet/)</b>|<b>[VGG](models/vgg/)</b>|<b>[Resnet](models/resnet/)</b>|
|A light-weight CNN providing Alexnet <br />level accuracy with 50X fewer<br /> parameters. Top-5 error from <br /> paper - ~20%|Deep CNN model (upto 19 layers)<br /> which won Imagenet Challenge in<br /> 2014. Top-5 error from <br /> paper - ~8%|Very deep CNN model (upto <br />152 layers) which won Imagenet<br /> Challenge in 2015. Top-5 error from <br /> paper - ~6%|
|<b>[Mobilenet](models/mobilenet/)</b>|||
|Efficient CNN model for mobile <br />and embedded vision applications.<br />Top-5 error from paper - ~10%|||
<!--
|[ONNX model]() (5 MB)<br /> [MMS archive]() <br />[Example notebook]()&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br />[Reference](https://arxiv.org/abs/1602.07360) <br />[Training notebook]() <br />Dataset - [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)|[ONNX model]() (*size*)<br /> [MMS archive]() <br />[Example notebook]()&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br />[Reference](https://arxiv.org/abs/1409.1556) <br />[Training notebook]() <br />Dataset - [ILSVRC2014](http://www.image-net.org/challenges/LSVRC/2014/) |[ONNX model]() (*size*)<br /> [MMS archive]() <br />[Example notebook]()&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br />[Reference](https://arxiv.org/abs/1512.03385) <br />[Training notebook]() <br />Dataset - [ILSVRC2015](http://www.image-net.org/challenges/LSVRC/2015/)
|<b>Mobilenet</b><br />*description*|<b>Densenet</b><br />*description*||
|[ONNX model]()<br /> [MMS archive]() <br />[Example notebook]()&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br />[Reference](https://arxiv.org/abs/1602.07360) <br />[Training notebook]() <br />Dataset - [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)|[ONNX model]()<br /> [MMS archive]() <br />[Example notebook]() &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br />[Reference](https://arxiv.org/abs/1602.07360) <br />[Training notebook]() <br />Dataset - [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)|
-->

### Face Detection and Recognition
<!--
The model takes images as input and detects/recognizes human faces in the image

| | | |
|-|-|-|
|<b>ArcFace</b><br /> *description*|||
|[ONNX model]()&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<br /> [MMS archive]() <br />[Example notebook]()<br />[Reference](https://arxiv.org/abs/1801.07698) <br />[Training notebook]() <br />[Dataset]() |
-->

### Semantic Segmentation
