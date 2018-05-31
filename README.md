<p align="center">
<img src="images/ONNX Model Zoo Graphics.png" width="90%"/>
</p>

# ONNX Model Zoo

This repository contains a collection of pre-trained models for state of the art works in deep learning. Each model is available in ONNX and MMS archive format. Accompanying each model are notebooks for training and running inference on the model written in MXNet framework, along with links to the dataset and the original paper.

Visualize the network architecture of a model using [Netron](https://lutzroeder.github.io/Netron).

Want to contribute a model? Check out the list of [backlog models](backlogs.md) to get started. Also refer to the [guidelines](contribute.md) for contribution before submitting a request.

*insert tags*
<!--
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
-->

# Models
## Image Classification
The model takes images as input and classifies the major object in the image into a set of pre-defined classes.

| | | |
|-|-|-|
|<b>[Squeezenet](models/squeezenet/squeezenet.md)</b>|<b>[VGG](models/vgg/vgg.md)</b>|<b>[Resnet](models/resnet/resnet.md)</b>|
|A light-weight CNN providing Alexnet <br />level accuracy with 50X fewer<br /> parameters|16 layer version of the deep CNN<br /> model which won Imagenet <br />Challenge in 2014|18 layer version of the deep CNN<br /> model which won Imagenet <br />Challenge in 2015|
|<b>[Mobilenet](models/mobilenet/mobilenet.md)</b>|||
|*description*|||
<!--
|[ONNX model]() (5 MB)<br /> [MMS archive]() <br />[Example notebook]()&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br />[Reference](https://arxiv.org/abs/1602.07360) <br />[Training notebook]() <br />Dataset - [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)|[ONNX model]() (*size*)<br /> [MMS archive]() <br />[Example notebook]()&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br />[Reference](https://arxiv.org/abs/1409.1556) <br />[Training notebook]() <br />Dataset - [ILSVRC2014](http://www.image-net.org/challenges/LSVRC/2014/) |[ONNX model]() (*size*)<br /> [MMS archive]() <br />[Example notebook]()&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br />[Reference](https://arxiv.org/abs/1512.03385) <br />[Training notebook]() <br />Dataset - [ILSVRC2015](http://www.image-net.org/challenges/LSVRC/2015/)
|<b>Mobilenet</b><br />*description*|<b>Densenet</b><br />*description*||
|[ONNX model]()<br /> [MMS archive]() <br />[Example notebook]()&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br />[Reference](https://arxiv.org/abs/1602.07360) <br />[Training notebook]() <br />Dataset - [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)|[ONNX model]()<br /> [MMS archive]() <br />[Example notebook]() &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <br />[Reference](https://arxiv.org/abs/1602.07360) <br />[Training notebook]() <br />Dataset - [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)|
-->
## Object Detection
<!--
The model takes images as input and detects objects present in the image

| | | |
|-|-|-|
|<b>SSD: Single Shot Multi Detector</b><br /> *description*|||
|[ONNX model]()&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<br /> [MMS archive]() <br />[Example notebook]()<br />[Reference](https://arxiv.org/abs/1512.02325) <br />[Training notebook]() <br />[Dataset]() |
-->
## Face Detection and Recognition
<!--
The model takes images as input and detects/recognizes human faces in the image

| | | |
|-|-|-|
|<b>ArcFace</b><br /> *description*|||
|[ONNX model]()&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<br /> [MMS archive]() <br />[Example notebook]()<br />[Reference](https://arxiv.org/abs/1801.07698) <br />[Training notebook]() <br />[Dataset]() |
-->

## Object Detection and Segmentation

## Semantic Segmentation
