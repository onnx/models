<p align="center">
<img src="images/ONNX Model Zoo Graphics.png" width="90%"/>
</p>

# Open Neural Network eXchange (ONNX) Model Zoo

[![Generic badge](https://img.shields.io/badge/Build-Work_In_Progress-red.svg)](#) 
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](#)

The ONNX Model Zoo is a collection of pre-trained models for state of the art models in deep learning. Models are available in the ONNX protobuf format. Where available and supported, the models are also available in the  Model Server archive format. Accompanying each model are [Jupyter](http://jupyter.org) notebooks for model training and running inference with the trained model. The notebooks are written in Python and include links to the training dataset as well as references to the original paper that describes the model architecture. The notebooks can be exported as python
 and run as python(.py) files.

## What is ONNX?
The Open Neural Network eXchange ([ONNX](http://onnx.ai)) is a open format to represent deep learning models. With ONNX, AI developers can more easily move models between state-of-the-art tools and choose the combination that is best for them. ONNX is developed and supported by a community of partners.
## Models

### Image Classification
This collection of models take images as input, then classifies the major objects in the images into a set of predefined classes.

|Model |Reference |Description |
|-|-|-|
|<b>[MobileNet](models/image_classification/mobilenet/)</b>|[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)|Efficient CNN model for mobile and embedded vision applications.Top-5 error from paper - ~10%|
|<b>[ResNet](models/image_classification/resnet/)</b>|[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)|Very deep CNN model (upto 152 layers) which won the ImageNet Challenge in 2015. Top-5 error from  paper - ~6%|
|<b>[SqueezeNet](models/image_classification/squeezenet/)</b>|[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)|A light-weight CNN providing Alexnet level accuracy with 50X fewer parameters. Top-5 error from  paper - ~20%|
|<b>[VGG](models/image_classification/vgg/)</b>|[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)|Deep CNN model (upto 19 layers) which won the ImageNet Challenge in 2014. Top-5 error from  paper - ~8%|
<hr>

### Face Detection
These models detect the presence of faces in images. Some more popular models are used for detection of celebrity faces, gender, age, and emotions.

|Model |Reference |Description |
|-|-|-|
|<b>ArcFace</b>|[ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)|[contribute](contribute.md)|
|<b>CNN Cascade</b>|[A Convolutional Neural Network Cascade for Face Detection ](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf)|[contribute](contribute.md)|

<hr>

### Object Detection & Segmentation
These models detect the presence of multiple objects in an image and segment out areas of the image where the objects are detected.

|Model |Reference |Description |
|-|-|-|
|<b>SSD</b>|[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)|[contribute](contribute.md)|
|<b>Faster-RCNN</b>|[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)|[contribute](contribute.md)|
|<b>Mask-RCNN</b>|[Mask R-CNN](https://arxiv.org/abs/1703.06870)|[contribute](contribute.md)|
|<b>YOLO v2</b>|[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)|[contribute](contribute.md)|
|<b>YOLO v3</b>|[YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)|[contribute](contribute.md)|

<hr>

### Semantic Segmentation
Semantic segmentation models will identify multiple classes of objects in an image and provide information on the areas of the image that object was detected.

|Model |Reference |Description |
|-|-|-|
|<b>FCN</b>|[Fully Convolutional Networks for Semantic Segmentation ](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=5&ved=0ahUKEwiI9YjW5ZDaAhVW5mMKHYifAL4QFghUMAQ&url=https%3A%2F%2Fpeople.eecs.berkeley.edu%2F~jonlong%2Flong_shelhamer_fcn.pdf&usg=AOvVaw1OZIT1dpO9NAS45hB7mhG8)|[contribute](contribute.md)|

<hr>

### Other CNN models

|Model |Reference |Description |
|-|-|-|
|   Gender Detection| [Age and Gender Classification using Convolutional Neural Networks](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=7&ved=0ahUKEwjEzsev5pDaAhVI2GMKHZvjCjEQFgiLATAG&url=http%3A%2F%2Fciteseerx.ist.psu.edu%2Fviewdoc%2Fdownload%3Fdoi%3D10.1.1.722.9654%26rep%3Drep1%26type%3Dpdf&usg=AOvVaw0-c9n2_ZcsRCyc6BCb0Zdj)	|[contribute](contribute.md)|
|	Super Resolution|	[Image Super resolution using deep convolutional networks ](http://ieeexplore.ieee.org/document/7115171/?reload=true)	|[contribute](contribute.md)|
<hr>

### GAN models
|Model |Reference |Description |
|-|-|-|
|	Text to Image|	[Generative Adversarial Text to image Synthesis ](https://arxiv.org/abs/1605.05396)|[contribute](contribute.md)|
|Style Transfer	|[Unpaired Image to Image Translation using Cycle consistent Adversarial Network ](https://arxiv.org/abs/1703.10593)|[contribute](contribute.md)|
|Sound Generative models|	[WaveNet: A Generative Model for Raw Audio ](https://arxiv.org/abs/1609.03499)|[contribute](contribute.md)|
<hr>

### NLP models

|Model |Reference |Description |
|-|-|-|
|Speech Recognition|	[Speech recognition with deep recurrent neural networks ](https://www.cs.toronto.edu/~fritz/absps/RNN13.pdf)|[contribute](contribute.md)|
|	Text To Speech|	[Deep voice: Real time neural text to speech ](https://arxiv.org/abs/1702.07825)	|[contribute](contribute.md)|
|	Language Model|	[Deep Neural Network Language Models ](https://pdfs.semanticscholar.org/a177/45f1d7045636577bcd5d513620df5860e9e5.pdf)	|[contribute](contribute.md)|
|	Machine Translation|	[Neural Machine Translation by jointly learning to align and translate ](https://arxiv.org/abs/1409.0473)|[contribute](contribute.md)|
|Machine Translation|	[Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation ](https://arxiv.org/abs/1609.08144)|[contribute](contribute.md)|
<hr>

### Models using Vision & NLP

|Model |Reference |Description |
|-|-|-|
|Visual Question Answering	|[VQA: Visual Question Answering ](https://arxiv.org/pdf/1505.00468v6.pdf)|[contribute](contribute.md)|
|Visual Question Answering	|[Yin and Yang: Balancing and Answering Binary Visual Questions ](https://arxiv.org/pdf/1511.05099.pdf)|[contribute](contribute.md)|
|Visual Question Answering	|[Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering](https://arxiv.org/pdf/1612.00837.pdf)|[contribute](contribute.md)|
|	Visual Dialog|	[Visual Dialog ](https://arxiv.org/abs/1611.08669)|[contribute](contribute.md)|

<hr>

### Other interesting models

|Model |Reference |Description |
|-|-|-|
|Time Series Forecasting|	[Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks ](https://arxiv.org/pdf/1703.07015.pdf)|[contribute](contribute.md)|
|Recommender systems|[DropoutNet: Addressing Cold Start in Recommender Systems](http://www.cs.toronto.edu/~mvolkovs/nips2017_deepcf.pdf)|[contribute](contribute.md)|
|Collaborative filtering||[contribute](contribute.md)|
|Autoencoders||[contribute](contribute.md)|

<hr>

## Model Serving
 Want to try out models instantly? Many of the models in this model zoo can be served with [Model Server](https://github.com/awslabs/mxnet-model-server) using the model archives provided. Model Server is a flexible and easy tool to serve deep learning models by providing a REST API with an inference end point. Supported ONNX models such as those converted from Chainer, CNTK, MXNet, and PyTorch can be served with Model Server. To learn about ONNX model serving with Model Server, refer to the [Model Server ONNX documentation](https://github.com/awslabs/mxnet-model-server/blob/master/docs/export_from_onnx.md). 
## Model Visualization
You can see visualizations of each model's network architecture by using [Netron](https://lutzroeder.github.io/Netron).

## Contributions
Do you want to contribute a model? To get started, pick any model presented above with the [contribute](contribute.md) link under the Description column. The links point to a page containing guidelines for making a contribution.
