
<!--
Title: Awesome Core ML Models
Description: A curated list of machine learning models in Core ML format.
Author: Kedan Li
-->
<p align="center">
<img src="images/coreml.png" width="329" height="295"/>
</p>

# Awesome Core ML Models

Since iOS 11, Apple released Core ML framework to help developers integrate machine learning models into applications. [The official documentation](https://developer.apple.com/documentation/coreml)

We've put up the largest collection of machine learning models in Core ML format, to help  iOS, macOS, tvOS, and watchOS developers experiment with machine learning techniques. We've created a site with better visualization of the models [CoreML.Store](https://coreml.store), and are working on more advance features.

If you've converted a Core ML model, feel free to submit an [issue](https://github.com/likedan/Awesome-CoreML-Models/issues/new).

Recently, we've included visualization tools. And here's one [Netron](https://lutzroeder.github.io/Netron).

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

# Models

## New Models
*Models that are recently added.*

| | | |
|-|-|-|
|[<img src="samples/cover_DocumentClassification.jpg">](https://coreml.store/documentclassification)|[<img src="http://via.placeholder.com/552x486/fafafa/dddddd/?text=great%20model%20to%20come">](https://coreml.store)|[<img src="http://via.placeholder.com/552x486/fafafa/dddddd/?text=great%20model%20to%20come">](https://coreml.store)|
|<b>DocumentClassification</b><br />Classify news articles into 1 of 5 categories.<br />[Download](https://coreml.store/documentclassification?download) \| [Demo](https://github.com/toddkramer/DocumentClassifier) \| [Reference](https://github.com/toddkramer/DocumentClassifier/)|||


## Image Processing
*Models that takes image data as input and output useful information about the image.*

| | | |
|-|-|-|
|[<img src="samples/cover_MobileNet.jpg">](https://coreml.store/mobilenet)|[<img src="samples/cover_GoogLeNetPlaces.jpg">](https://coreml.store/googlenetplaces)|[<img src="samples/cover_Inceptionv3.jpg">](https://coreml.store/inceptionv3)|
|<b>MobileNet</b><br />The network from the paper \'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications\', trained on the ImageNet dataset.<br />[Download](https://coreml.store/mobilenet?download) \| [Demo](https://github.com/hollance/MobileNet-CoreML) \| [Reference](https://arxiv.org/abs/1704.04861)|<b>GoogLeNetPlaces</b><br />Detects the scene of an image from 205 categories such as airport, bedroom, forest, coast etc.<br />[Download](https://coreml.store/googlenetplaces?download) \| [Demo](https://github.com/chenyi1989/CoreMLDemo) \| [Reference](http://places.csail.mit.edu/index.html)|<b>Inceptionv3</b><br />Detects the dominant objects present in an image from a set of 1000 categories such as trees, animals, food, vehicles, person etc. The top-5 error from the original publication is 5.6%.<br />[Download](https://coreml.store/inceptionv3?download) \| [Demo](https://github.com/yulingtianxia/Core-ML-Sample/) \| [Reference](https://arxiv.org/abs/1512.00567)|
|[<img src="samples/cover_Resnet50.jpg">](https://coreml.store/resnet50)|[<img src="samples/cover_VGG16.jpg">](https://coreml.store/vgg16)|[<img src="samples/cover_CarRecognition.jpg">](https://coreml.store/carrecognition)|
|<b>Resnet50</b><br />Detects the dominant objects present in an image from a set of 1000 categories such as trees, animals, food, vehicles, person etc. The top-5 error from the original publication is 7.8%.<br />[Download](https://coreml.store/resnet50?download) \| [Demo](https://github.com/atomic14/VisionCoreMLSample) \| [Reference](https://arxiv.org/abs/1512.03385)|<b>VGG16</b><br />Detects the dominant objects present in an image from a set of 1000 categories such as trees, animals, food, vehicles, person etc. The top-5 error from the original publication is 7.4%.<br />[Download](https://coreml.store/vgg16?download) \| [Demo](https://github.com/alaphao/CoreMLExample) \| [Reference](https://arxiv.org/abs/1409.1556)|<b>CarRecognition</b><br />Predict the brand & model of a car.<br />[Download](https://coreml.store/carrecognition?download) \| [Demo](https://github.com/likedan/Core-ML-Car-Recognition) \| [Reference](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)|
|[<img src="samples/cover_TinyYOLO.jpg">](https://coreml.store/tinyyolo)|[<img src="samples/cover_AgeNet.jpg">](https://coreml.store/agenet)|[<img src="samples/cover_GenderNet.jpg">](https://coreml.store/gendernet)|
|<b>TinyYOLO</b><br />The Tiny YOLO network from the paper \'YOLO9000: Better, Faster, Stronger\' (2016), arXiv:1612.08242<br />[Download](https://coreml.store/tinyyolo?download) \| [Demo](https://github.com/hollance/YOLO-CoreML-MPSNNGraph) \| [Reference](http://machinethink.net/blog/object-detection-with-yolo)|<b>AgeNet</b><br />Age Classification using Convolutional Neural Networks<br />[Download](https://coreml.store/agenet?download) \| [Demo](https://github.com/cocoa-ai/FacesVisionDemo) \| [Reference](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/)|<b>GenderNet</b><br />Gender Classification using Convolutional Neural Networks<br />[Download](https://coreml.store/gendernet?download) \| [Demo](https://github.com/cocoa-ai/FacesVisionDemo) \| [Reference](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/)|
|[<img src="samples/cover_MNIST.jpg">](https://coreml.store/mnist)|[<img src="samples/cover_CNNEmotions.jpg">](https://coreml.store/cnnemotions)|[<img src="samples/cover_VisualSentimentCNN.jpg">](https://coreml.store/visualsentimentcnn)|
|<b>MNIST</b><br />Predicts a handwritten digit.<br />[Download](https://coreml.store/mnist?download) \| [Demo](https://github.com/ph1ps/MNIST-CoreML) \| [Reference](http://yann.lecun.com/exdb/mnist/)|<b>CNNEmotions</b><br />Emotion Recognition in the Wild via Convolutional Neural Networks and Mapped Binary Patterns<br />[Download](https://coreml.store/cnnemotions?download) \| [Demo](https://github.com/cocoa-ai/FacesVisionDemo) \| [Reference](http://www.openu.ac.il/home/hassner/projects/cnn_emotions/)|<b>VisualSentimentCNN</b><br />Fine-tuning CNNs for Visual Sentiment Prediction<br />[Download](https://coreml.store/visualsentimentcnn?download) \| [Demo](https://github.com/cocoa-ai/SentimentVisionDemo) \| [Reference](http://www.sciencedirect.com/science/article/pii/S0262885617300355?via%3Dihub)|
|[<img src="samples/cover_Food101.jpg">](https://coreml.store/food101)|[<img src="samples/cover_Oxford102.jpg">](https://coreml.store/oxford102)|[<img src="samples/cover_FlickrStyle.jpg">](https://coreml.store/flickrstyle)|
|<b>Food101</b><br />This model takes a picture of a food and predicts its name<br />[Download](https://coreml.store/food101?download) \| [Demo](https://github.com/ph1ps/Food101-CoreML) \| [Reference](http://visiir.lip6.fr/explore)|<b>Oxford102</b><br />Classifying images in the Oxford 102 flower dataset with CNNs<br />[Download](https://coreml.store/oxford102?download) \| [Demo](https://github.com/cocoa-ai/FlowersVisionDemo) \| [Reference](http://jimgoo.com/flower-power/)|<b>FlickrStyle</b><br />Finetuning CaffeNet on Flickr Style<br />[Download](https://coreml.store/flickrstyle?download) \| [Demo](https://github.com/cocoa-ai/StylesVisionDemo) \| [Reference](http://sergeykarayev.com/files/1311.3715v3.pdf)|
|[<img src="samples/cover_RN1015k500.jpg">](https://coreml.store/rn1015k500)|[<img src="samples/cover_Nudity.jpg">](https://coreml.store/nudity)|[<img src="http://via.placeholder.com/552x486/fafafa/dddddd/?text=great%20model%20to%20come">](https://coreml.store)|
|<b>RN1015k500</b><br />Predict the location where a picture was taken.<br />[Download](https://coreml.store/rn1015k500?download) \| [Demo](https://github.com/awslabs/MXNet2CoreML_iOS_sample_app) \| [Reference](https://aws.amazon.com/blogs/ai/estimating-the-location-of-images-using-mxnet-and-multimedia-commons-dataset-on-aws-ec2)|<b>Nudity</b><br />Classifies an image either as NSFW (nude) or SFW (not nude)<br />[Download](https://coreml.store/nudity?download) \| [Demo](https://github.com/ph1ps/Nudity-CoreML) \| [Reference](https://github.com/yahoo/open_nsfw)||

## Style Transfer
*Models that transform image to specific style.*

| | | |
|-|-|-|
|[<img src="samples/cover_HED_so.jpg">](https://coreml.store/hed_so)|[<img src="samples/cover_FNS-Candy.jpg">](https://coreml.store/fns-candy)|[<img src="samples/cover_FNS-Feathers.jpg">](https://coreml.store/fns-feathers)|
|<b>HED_so</b><br />Holistically-Nested Edge Detection. Side outputs<br />[Download](https://coreml.store/hed_so?download) \| [Demo](https://github.com/s1ddok/HED-CoreML) \| [Reference](http://dl.acm.org/citation.cfm?id=2654889)|<b>FNS-Candy</b><br />Feedforward style transfer https://github.com/jcjohnson/fast-neural-style<br />[Download](https://coreml.store/fns-candy?download) \| [Demo](https://github.com/prisma-ai/torch2coreml) \| [Reference](http://cs.stanford.edu/people/jcjohns/eccv16/)|<b>FNS-Feathers</b><br />Feedforward style transfer https://github.com/jcjohnson/fast-neural-style<br />[Download](https://coreml.store/fns-feathers?download) \| [Demo](https://github.com/prisma-ai/torch2coreml) \| [Reference](http://cs.stanford.edu/people/jcjohns/eccv16/)|
|[<img src="samples/cover_FNS-La-Muse.jpg">](https://coreml.store/fns-la-muse)|[<img src="samples/cover_FNS-The-Scream.jpg">](https://coreml.store/fns-the-scream)|[<img src="samples/cover_FNS-Udnie.jpg">](https://coreml.store/fns-udnie)|
|<b>FNS-La-Muse</b><br />Feedforward style transfer https://github.com/jcjohnson/fast-neural-style<br />[Download](https://coreml.store/fns-la-muse?download) \| [Demo](https://github.com/prisma-ai/torch2coreml) \| [Reference](http://cs.stanford.edu/people/jcjohns/eccv16/)|<b>FNS-The-Scream</b><br />Feedforward style transfer https://github.com/jcjohnson/fast-neural-style<br />[Download](https://coreml.store/fns-the-scream?download) \| [Demo](https://github.com/prisma-ai/torch2coreml) \| [Reference](http://cs.stanford.edu/people/jcjohns/eccv16/)|<b>FNS-Udnie</b><br />Feedforward style transfer https://github.com/jcjohnson/fast-neural-style<br />[Download](https://coreml.store/fns-udnie?download) \| [Demo](https://github.com/prisma-ai/torch2coreml) \| [Reference](http://cs.stanford.edu/people/jcjohns/eccv16/)|
|[<img src="samples/cover_FNS-Mosaic.jpg">](https://coreml.store/fns-mosaic)|[<img src="samples/cover_AnimeScale2x.jpg">](https://coreml.store/animescale2x)|[<img src="http://via.placeholder.com/552x486/fafafa/dddddd/?text=great%20model%20to%20come">](https://coreml.store)|
|<b>FNS-Mosaic</b><br />Feedforward style transfer https://github.com/jcjohnson/fast-neural-style<br />[Download](https://coreml.store/fns-mosaic?download) \| [Demo](https://github.com/prisma-ai/torch2coreml) \| [Reference](http://cs.stanford.edu/people/jcjohns/eccv16/)|<b>AnimeScale2x</b><br />Process a bicubic-scaled anime-style artwork<br />[Download](https://coreml.store/animescale2x?download) \| [Demo](https://github.com/imxieyi/waifu2x-ios) \| [Reference](https://arxiv.org/abs/1501.00092)||

## Text Analysis
*Models that takes text data as input and output useful information about the text.*

| | | |
|-|-|-|
|[<img src="samples/cover_SentimentPolarity.jpg">](https://coreml.store/sentimentpolarity)|[<img src="samples/cover_DocumentClassification.jpg">](https://coreml.store/documentclassification)|[<img src="samples/cover_MessageClassifier.jpg">](https://coreml.store/messageclassifier)|
|<b>SentimentPolarity</b><br />Sentiment polarity LinearSVC.<br />[Download](https://coreml.store/sentimentpolarity?download) \| [Demo](https://github.com/cocoa-ai/SentimentCoreMLDemo) \| [Reference](http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW3/)|<b>DocumentClassification</b><br />Classify news articles into 1 of 5 categories.<br />[Download](https://coreml.store/documentclassification?download) \| [Demo](https://github.com/toddkramer/DocumentClassifier) \| [Reference](https://github.com/toddkramer/DocumentClassifier/)|<b>MessageClassifier</b><br />Detect whether a message is spam.<br />[Download](https://coreml.store/messageclassifier?download) \| [Demo](https://github.com/gkswamy98/imessage-spam-detection/tree/master) \| [Reference](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)|
|[<img src="samples/cover_NamesDT.jpg">](https://coreml.store/namesdt)|[<img src="http://via.placeholder.com/552x486/fafafa/dddddd/?text=great%20model%20to%20come">](https://coreml.store)|[<img src="http://via.placeholder.com/552x486/fafafa/dddddd/?text=great%20model%20to%20come">](https://coreml.store)|
|<b>NamesDT</b><br />Gender Classification using DecisionTreeClassifier<br />[Download](https://coreml.store/namesdt?download) \| [Demo](https://github.com/cocoa-ai/NamesCoreMLDemo) \| [Reference](http://nlpforhackers.io)|||

## Others

| | | |
|-|-|-|
|[<img src="samples/cover_Exermote.jpg">](https://coreml.store/exermote)|[<img src="samples/cover_GestureAI.jpg">](https://coreml.store/gestureai)|[<img src="http://via.placeholder.com/552x486/fafafa/dddddd/?text=great%20model%20to%20come">](https://coreml.store)|
|<b>Exermote</b><br />Predicts the exercise, when iPhone is worn on right upper arm.<br />[Download](https://coreml.store/exermote?download) \| [Demo](https://github.com/Lausbert/Exermote/tree/master/ExermoteInference) \| [Reference](http://lausbert.com/2017/08/03/exermote/)|<b>GestureAI</b><br />GestureAI<br />[Download](https://coreml.store/gestureai?download) \| [Demo](https://github.com/akimach/GestureAI-iOS/tree/master/GestureAI) \| [Reference](https://github.com/akimach/GestureAI-iOS/tree/master/GestureAI)||


# Visualization Tools
*Tools that helps visualize CoreML Models*
* [Netron](https://lutzroeder.github.io/Netron)

# Supported formats
*List of model formats that could be converted to Core ML with examples*
* [Caffe](https://apple.github.io/coremltools/generated/coremltools.converters.caffe.convert.html)
* [Keras](https://apple.github.io/coremltools/generated/coremltools.converters.keras.convert.html)
* [XGBoost](https://apple.github.io/coremltools/generated/coremltools.converters.xgboost.convert.html)
* [Scikit-learn](https://apple.github.io/coremltools/generated/coremltools.converters.sklearn.convert.html)
* [MXNet](https://aws.amazon.com/blogs/ai/bring-machine-learning-to-ios-apps-using-apache-mxnet-and-apple-core-ml/)
* [LibSVM](https://apple.github.io/coremltools/generated/coremltools.converters.libsvm.convert.html)
* [Torch7](https://github.com/prisma-ai/torch2coreml)

# The Gold
*Collections of machine learning models that could be converted to Core ML*

* [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) - Big list of models in Caffe format.
* [TensorFlow Models](https://github.com/tensorflow/models) - Models for TensorFlow.
* [TensorFlow Slim Models](https://github.com/tensorflow/models/blob/master/slim/README.md) - Another collection of TensorFlow Models.
* [MXNet Model Zoo](https://mxnet.incubator.apache.org/model_zoo/) - Collection of MXNet models.

*Individual machine learning models that could be converted to Core ML. We'll keep adjusting the list as they become converted.*
* [LaMem](https://github.com/MiyainNYC/Visual-Memorability-through-Caffe) Score the memorability of pictures.
* [ILGnet](https://github.com/BestiVictory/ILGnet) The aesthetic evaluation of images.
* [Colorization](https://github.com/richzhang/colorization) Automatic colorization using deep neural networks.
* [Illustration2Vec](https://github.com/rezoo/illustration2vec) Estimating a set of tags and extracting semantic feature vectors from given illustrations.
* [CTPN](https://github.com/tianzhi0549/CTPN) Detecting text in natural image.
* [Image Analogy](https://github.com/msracver/Deep-Image-Analogy) Find semantically-meaningful dense correspondences between two input images.
* [iLID](https://github.com/twerkmeister/iLID) Automatic spoken language identification.
* [Fashion Detection](https://github.com/liuziwei7/fashion-detection) Cloth detection from images.
* [Saliency](https://github.com/imatge-upc/saliency-2016-cvpr) The prediction of salient areas in images has been traditionally addressed with hand-crafted features.
* [Face Detection](https://github.com/DolotovEvgeniy/DeepPyramid) Detect face from image.
* [mtcnn](https://github.com/CongWeilin/mtcnn-caffe) Joint Face Detection and Alignment.
* [deephorizon](https://github.com/scottworkman/deephorizon) Single image horizon line estimation.

# Contributing and License
* [See the guide](https://github.com/likedan/Awesome-CoreML-Models/blob/master/.github/CONTRIBUTING.md)
* Distributed under the MIT license. See LICENSE for more information.
