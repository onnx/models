# Open Neural Network eXchange (ONNX) Model Zoo
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](contribute.md)

<p align="center">
	<img src="resource/images/ONNX_Model_Zoo_Graphics.png" width="60%"/>
</p>

The ONNX Model Zoo is a collection of pre-trained models for state-of-the-art models in deep learning, available in the ONNX format. Accompanying each model are [Jupyter notebooks](http://jupyter.org) for model training and running inference with the trained model. The notebooks are written in Python and include links to the training dataset as well as references to the original paper that describes the model architecture. The notebooks can also be exported and run as Python (.py) files.

## What is ONNX?
The Open Neural Network eXchange ([ONNX](http://onnx.ai)) is an open format to represent deep learning models. With ONNX, developers can move models between state-of-the-art tools and choose the combination that is best for them. ONNX is developed and supported by a community of partners.

## Models
#### Read the [Usage](#usage-) section below for more details on the file formats in the ONNX Model Zoo (.onnx, .pb, .npz) and starter Python code for validating your ONNX model using test data.

* [Image Classification](#image_classification)
* [Object Detection & Image Segmentation](#object_detection)
* [Body, Face & Gesture Analysis](#body_analysis)
* [Image Manipulation](#image_manipulation)
* [Speech & Audio Processing](#speech)
* [Machine Comprehension](#machine_comprehension)
* [Machine Translation](#machine_translation)
* [Language Modelling](#language)
* [Visual Question Answering & Dialog](#visual_qna)
* [Other interesting models](#others)

### Image Classification <a name="image_classification"/>
This collection of models take images as input, then classifies the major objects in the images into a set of predefined classes.

|Model Class |Reference |Description |
|-|-|-|
<<<<<<< HEAD
|<b>[MobileNet](models/image_classification/mobilenet/)</b>|[Sandler et al.](https://arxiv.org/abs/1801.04381)|CNN model that increases efficiency in performance. Used for mobile and embedded vision applications.|
|<b>[ResNet](models/image_classification/resnet/)</b>|[He et al.](https://arxiv.org/abs/1512.03385), [He et al.](https://arxiv.org/abs/1603.05027)|Very deep state-of-the-art CNN model (up to 152 layers). This model accelerates the speed of training of deep networks and solves the vanishing gradient problem by using shorcut connections. |
|<b>[SqueezeNet](models/image_classification/squeezenet/)</b>|[Iandola et al.](https://arxiv.org/abs/1602.07360)|A light-weight CNN providing Alexnet level accuracy with 50X fewer parameters. This model is easy to deploy and requires less communication across servers.|
|<b>[VGG](models/image_classification/vgg/)</b>|[Simonyan et al.](https://arxiv.org/abs/1409.1556)|Deep CNN model(up to 19 layers). Provides high accuracy but at cost of large model size.|
|<b>[Bvlc_AlexNet](bvlc_alexnet)</b>|[Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)|A Deep CNN model (up to 8 layers) with fast speed. Classifies into over a thousand categories for a wide range of images.|
|<b>[Bvlc_GoogleNet](bvlc_googlenet)</b>|[Szegedy et al.](https://arxiv.org/pdf/1409.4842.pdf)|Deep CNN model(up to 22 layers). Comparatively smaller and faster than VGG and more accurate in detailing than AlexNet.|
|<b>[Bvlc_reference_CaffeNet](bvlc_reference_caffenet)</b>|[Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)|Deep CNN variation of AlexNet for Image Classification in Caffe where the max pooling precedes the local response normalization (LRN) so that the LRN takes less compute and memory.|
|<b>[Bvlc_reference_RCNN_ILSVRC13](bvlc_reference_rcnn_ilsvrc13)</b>|[Girshick et al.](https://arxiv.org/abs/1311.2524)|Pure Caffe implementation of R-CNN for image classification. This model uses localization of regions to classify and extract features from images.|
|<b>[DenseNet121](densenet121)</b>|[Huang et al.](https://arxiv.org/abs/1608.06993)|Model that has every layer connected to every other layer and passes on its own feature providing strong gradient flow and more diversified features.|
|<b>[Inception_v1](inception_v1)</b>|[Szegedy et al.](https://arxiv.org/abs/1409.4842)|This model is same as GoogLeNet, implemented through Caffe2 that has improved utilization of the computing resources inside the network and helps with the vanishing gradient problem.|
|<b>[Inception_v2](inception_v2)</b>|[Szegedy et al.](https://arxiv.org/abs/1512.00567)|CNN model for that is an adaptation to Inception v1 with batch normalization. This model reduces the dimensions too much that may causes loss of information thus providing greater accuracy when classifying images.|
|<b>[ShuffleNet](shufflenet)</b>|[Zhang et al.](https://arxiv.org/abs/1707.01083)|Extremely computation efficient CNN model that is designed specifically for mobile devices. This model greatly reduces the computational cost and provides a ~13x speedup over AlexNet on ARM-based mobile devices.|
|<b>[ZFNet512](zfnet512)</b>|[Zeiler et al.](https://arxiv.org/abs/1311.2901)|Deep CNN model (up to 8 layers) that increased the number of features that the network is capable of detecting that helps to pick image features at a finer level of resolution.|
=======
|<b>[MobileNet](vision/classification/mobilenet)</b>|[Sandler et al.](https://arxiv.org/abs/1801.04381)|Computationally efficient CNN model for mobile and embedded vision applications. <br>Top-5 error from paper - ~10%|
|<b>[ResNet](vision/classification/resnet)</b>|[He et al.](https://arxiv.org/abs/1512.03385), [He et al.](https://arxiv.org/abs/1603.05027)|Very deep state-of-the-art CNN model (up to 152 layers), won the ImageNet Challenge in 2015. <br>Top-5 error from  paper - ~3.6%|
|<b>[SqueezeNet](vision/classification/squeezenet)</b>|[Iandola et al.](https://arxiv.org/abs/1602.07360)|A light-weight CNN providing Alexnet level accuracy with 50X fewer parameters. <br>Top-5 error from  paper - ~20%|
|<b>[VGG](vision/classification/vgg)</b>|[Simonyan et al.](https://arxiv.org/abs/1409.1556)|Deep CNN model (up to 19 layers) which won the ImageNet Challenge in 2014. <br>Top-5 error from  paper - ~8%|
|<b>[AlexNet](vision/classification/alexnet)</b>|[Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)|Deep CNN model for Image Classification (up to 8 layers), won the ImageNet Challenge in 2012. <br>Top-5 error from paper - ~15%|
|<b>[GoogleNet](vision/classification/inception_and_googlenet/googlenet)</b>|[Szegedy et al.](https://arxiv.org/pdf/1409.4842.pdf)|Deep CNN model (up to 22 layers) implemented in Caffe and won at the ImageNet Challenge in 2014. <br>Top-5 error from paper - ~6.7%|
|<b>[CaffeNet](vision/classification/caffenet)</b>|[Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)|Deep CNN variation of AlexNet for Image Classification in Caffe where the max pooling precedes the local response normalization (LRN) so that the LRN takes less compute and memory.|
|<b>[RCNN_ILSVRC13](vision/classification/rcnn_ilsvrc13)</b>|[Girshick et al.](https://arxiv.org/abs/1311.2524)|Pure Caffe implementation of R-CNN for image classification as presented at CVPR in 2014.|
|<b>[DenseNet-121](vision/classification/densenet-121)</b>|[Huang et al.](https://arxiv.org/abs/1608.06993)|Deep CNN model for Image Classification, connecting every layer to every other layer.|
|<b>[Inception_V1](vision/classification/inception_and_googlenet/inception_v1)</b>|[Szegedy et al.](https://arxiv.org/abs/1409.4842)|Deep CNN model (up to 22 layers) for Image Classification - same as GoogLeNet, implemented through Caffe2. <br>Top-5 error from paper - ~6.7%|
|<b>[Inception_V2](vision/classification/inception_and_googlenet/inception_v2)</b>|[Szegedy et al.](https://arxiv.org/abs/1512.00567)|Deep CNN model for Image Classification as an adaptation to Inception v1 with batch normalization <br> Top-5 error from paper ~4.82%|
|<b>[ShuffleNet](vision/classification/shufflenet)</b>|[Zhang et al.](https://arxiv.org/abs/1707.01083)|Computationally efficient deep CNN model for Image Classification, providing a ~13x speedup over AlexNet on ARM-based mobile devices <br> Top-1 error from paper - ~7.8%|
|<b>[ZFNet-512](vision/classification/zfnet-512)</b>|[Zeiler et al.](https://arxiv.org/abs/1311.2901)|Deep CNN model (up to 8 layers) for Image Classification that tuned the hyperparameters of AlexNet and won the ImageNet Challenge in 2013. <br> Top-5 error from paper - ~14.3%|
>>>>>>> 8d50e3f598e6d5c67c7c7253e5a203a26e731a1b
<hr>

#### Domain-based Image Classification <a name="domain_based_image"/>
This subset of models classify images for specific domains and datasets.

|Model Class |Reference |Description |
|-|-|-|
|<b>[MNIST-Handwritten Digit Recognition](vision/classification/mnist)</b>|[Convolutional Neural Network with MNIST](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.ipynb)	|Deep CNN model for handwritten digit identification|
<hr>

### Object Detection & Image Segmentation <a name="object_detection"/>
Object detection models detect the presence of multiple objects in an image and segment out areas of the image where the objects are detected. Semantic segmentation models partition an input image by labeling each pixel into a set of pre-defined categories.

|Model Class |Reference |Description |
|-|-|-|
<<<<<<< HEAD
|<b>[Tiny_YOLOv2](tiny_yolov2)</b>|[Redmon et al.](https://arxiv.org/pdf/1612.08242.pdf)|A real-time CNN for object detection that detects 20 different classes. A smaller version of the more complex full YOLOv2 network.|
|<b>[SSD](ssd)</b>|[Liu et al.](https://arxiv.org/abs/1512.02325)|Single Stage Detector: real-time CNN for object detection that detects 80 different classes.|
|<b>[Faster-RCNN](faster_rcnn)</b>|[Ren et al.](https://arxiv.org/abs/1506.01497)|Increases efficiency from R-CNN by connecting a RPN with a CNN to create a single, unified network for object detection that detects 80 different classes.|
|<b>[Mask-RCNN](mask_rcnn)</b>|[He et al.](https://arxiv.org/abs/1703.06870)|A real-time neural network for object instance segmentation that detects 80 different classes. Extends Faster R-CNN as each of the 300 elected ROIs go through 3 parallel branches of the network: label prediction, bounding box prediction and mask prediction.|
|<b>YOLO v2</b>|[Redmon et al.](https://arxiv.org/abs/1612.08242)|A CNN model for real-time object detection system that can detect over 9000 object categories. It uses a single network evaluation, enabling it to be more than 1000x faster than R-CNN and 100x faster than Faster R-CNN. <br>[contribute](contribute.md)|
|<b>[YOLO v3](yolov3)</b>|[Redmon et al.](https://pjreddie.com/media/files/papers/YOLOv3.pdf)|A deep CNN model for real-time object detection that detects 80 different classes. A little bigger than YOLOv2 but still very fast. As accurate as SSD but 3 times faster.|
|<b>[DUC](models/semantic_segmentation/DUC/)</b>|[Wang et al.](https://arxiv.org/abs/1702.08502)|Deep CNN based pixel-wise semantic segmentation model with >80% [mIOU](/models/semantic_segmentation/DUC/README.md/#metric) (mean Intersection Over Union). Trained on cityscapes dataset, which can be effectively implemented in self driving vehicle systems.|
|<b>FCN</b>|[Long et al.](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)|Deep CNN based segmentation model trained end-to-end, pixel-to-pixel that produces efficient inference and learning. Built off of AlexNet, VGG net, GoogLeNet classification methods. <br>[contribute](contribute.md)|
=======
|<b>[Tiny YOLOv2](vision/object_detection_segmentation/tiny_yolov2)</b>|[Redmon et al.](https://arxiv.org/pdf/1612.08242.pdf)|Deep CNN model for Object Detection|
|<b>[SSD](vision/object_detection_segmentation/ssd)</b>|[Liu et al.](https://arxiv.org/abs/1512.02325)|Deep CNN model for Object Detection|
|<b>[Faster-RCNN](vision/object_detection_segmentation/faster-rcnn)</b>|[Ren et al.](https://arxiv.org/abs/1506.01497)|Deep CNN model for Object Detection|
|<b>[Mask-RCNN](vision/object_detection_segmentation/mask-rcnn)</b>|[He et al.](https://arxiv.org/abs/1703.06870)|Deep CNN model for Object Segmentation|
|<b>YOLO v2</b>|[Redmon et al.](https://arxiv.org/abs/1612.08242)|[contribute](contribute.md)|
|<b>[YOLO v3](vision/object_detection_segmentation/yolov3)</b>|[Redmon et al.](https://pjreddie.com/media/files/papers/YOLOv3.pdf)|Deep CNN model for Real-Time Object Detection (mAP = 55.3% in COCO)|
|<b>[DUC](vision/object_detection_segmentation/duc)</b>|[Wang et al.](https://arxiv.org/abs/1702.08502)|Deep CNN based semantic segmentation model with >80% [mIOU](/models/semantic_segmentation/DUC/README.md/#metric) (mean Intersection Over Union), trained on urban street images|
|<b>FCN</b>|[Long et al.](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)|[contribute](contribute.md)|
>>>>>>> 8d50e3f598e6d5c67c7c7253e5a203a26e731a1b
<hr>

### Body, Face & Gesture Analysis <a name="body_analysis"/>
Face detection models identify and/or recognize human faces in images. Some more popular models are used for detection of celebrity faces, gender, age, and emotions.

|Model Class |Reference |Description |
|-|-|-|
<<<<<<< HEAD
|<b>[ArcFace](models/face_recognition/ArcFace/)</b>|[Deng et al.](https://arxiv.org/abs/1801.07698)|Model for face recognition that discriminative features of faces and produces embeddings. It does not need to be combined with other loss functions in order to have stable performance, and it can easily converge on any training datasets.|
|<b>CNN Cascade</b>|[Li et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf)|The model operates at multiple resolutions, quickly rejecting the background regions in the fast low resolution stages in an image and carefully evaluates a small number of challenging candidates in the last high resolution stage.<br>[contribute](contribute.md)|
|[**Emotion FerPlus**](emotion_ferplus) |[Barsoum et al.](https://arxiv.org/abs/1608.01041)	|This model is trained using the FER+ dataset which is an enhanced data set with multiple labels for each face image. This dataset can be used to train a deep convolutional neural network (DCNN) from noisy labels, using facial expression recognition as an example. |
|Age and Gender Classification using Convolutional Neural Networks| [Levi et al.](https://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf)	|This model accurately classifies gender and age even the amount of learning data is limited.<br>[contribute](contribute.md)|
=======
|<b>[ArcFace](vision/body_analysis/arcface)</b>|[Deng et al.](https://arxiv.org/abs/1801.07698)|ArcFace is a CNN based model for face recognition which learns discriminative features of faces and produces embeddings for input face images.|
|<b>CNN Cascade</b>|[Li et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf)|[contribute](contribute.md)|
|[Emotion FerPlus](vision/body_analysis/emotion_ferplus) |[Barsoum et al.](https://arxiv.org/abs/1608.01041)	| Deep CNN for emotion recognition trained on images of faces.|
|Age and Gender Classification using Convolutional Neural Networks| [Levi et al.](https://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf)	|[contribute](contribute.md)|
>>>>>>> 8d50e3f598e6d5c67c7c7253e5a203a26e731a1b
<hr>

### Image Manipulation <a name="image_manipulation"/>
Image manipulation models use neural networks to transform input images to modified output images. Some popular models in this category involve style transfer or enhancing images by increasing resolution.

|Model Class |Reference |Description |
|-|-|-|
|Unpaired Image to Image Translation using Cycle consistent Adversarial Network|[Zhu et al.](https://arxiv.org/abs/1703.10593)|The model uses learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. <br>[contribute](contribute.md)|
|Image Super resolution using deep convolutional networks |	[Dong et al.](http://ieeexplore.ieee.org/document/7115171/?reload=true)	|A deep CNN that takes low-resolution image as the input and outputs the high-resolution one. Fast speed for restoration quality. <br>[contribute](contribute.md)|
<hr>

### Speech & Audio Processing <a name="speech"/>
This class of models uses audio data to train models that can identify voice, generate music, or even read text out loud.

|Model Class |Reference |Description |
|-|-|-|
|Speech recognition with deep recurrent neural networks|	[Graves et al.](https://www.cs.toronto.edu/~fritz/absps/RNN13.pdf)|A RNN model for sequential data for speech recognition. Labels problems where the input-output alignment is unknown<br>[contribute](contribute.md)|
|Deep voice: Real time neural text to speech |	[Arik et al.](https://arxiv.org/abs/1702.07825)	|A DNN model performs end-to-end neural speech synthesis. Requires fewer parameters and its faster/more flexible than other systems. <br>[contribute](contribute.md)|
|Sound Generative models|	[WaveNet: A Generative Model for Raw Audio ](https://arxiv.org/abs/1609.03499)|A GAN that generates raw audio waveforms. Has predictive distribution for each audio sample. Generates realistic music fragments. <br>[contribute](contribute.md)|
<hr>

### Machine Comprehension <a name="machine_comprehension"/>
This subset of natural language processing models that answer questions about a given context paragraph.

|Model Class |Reference |Description |
|-|-|-|
<<<<<<< HEAD
|Bidirectional Attention Flow|[Seo et al.](https://arxiv.org/pdf/1611.01603)|A model that answers a query about a given context paragraph. This is a multi-stage hierarchical process that represents the context at different levels of granularity and uses bidirectional attention flow mechanism to obtain a query-aware context representation without early summarization.|
=======
|<b>[Bidirectional Attention Flow](text/machine_comprehension/bidirectional_attention_flow)</b>|[Seo et al.](https://arxiv.org/pdf/1611.01603)|EM of 68.1% in SQuADv1.1|
>>>>>>> 8d50e3f598e6d5c67c7c7253e5a203a26e731a1b
<hr>

### Machine Translation <a name="machine_translation"/>
This class of natural language processing models learns how to translate input text to another language.

|Model Class |Reference |Description |
|-|-|-|
|Neural Machine Translation by jointly learning to align and translate|	[Bahdanau et al.](https://arxiv.org/abs/1409.0473)|Aims to build a single neural network that can be jointly tuned to maximize the translation performance. The model belongs to a family of encoder-decoders and consists of an encoder that encodes a source sentence into a fixed-length vector from which a decoder generates a translation. <br>[contribute](contribute.md)|
|Google's Neural Machine Translation System|	[Wu et al.](https://arxiv.org/abs/1609.08144)|This model helps to improve issues faced by the Neural Machine Translation (NMT) systems like parallelism that helps to decrease training time, accelerate the final translation speed, and the handling of rare words dividing words into a limited set of common sub-word units for both input and output.<br>[contribute](contribute.md)|
<hr>

### Language Modelling <a name="language"/>
This subset of natural language processing models learns representations of language from large corpuses of text.

|Model Class |Reference |Description |
|-|-|-|
|Deep Neural Network Language Models | [Arisoy et al.](https://pdfs.semanticscholar.org/a177/45f1d7045636577bcd5d513620df5860e9e5.pdf)|A DNN acoustic model. Used in many natural language technologies. Represents a probability distribution over all possible word strings in a language. <br> [contribute](contribute.md)|
<hr>

### Visual Question Answering & Dialog <a name="visual_qna"/>
This subset of natural language processing models uses input images to answer questions about those images.

|Model Class |Reference |Description |
|-|-|-|
|VQA: Visual Question Answering |[Agrawal et al.](https://arxiv.org/pdf/1505.00468v6.pdf)|A model that takes an image and a free-form, open-ended natural language question about the image and outputs a natural-language answer. <br>[contribute](contribute.md)|
|Yin and Yang: Balancing and Answering Binary Visual Questions |[Zhang et al.](https://arxiv.org/pdf/1511.05099.pdf)|Addresses VQA by converting the question to a tuple that concisely summarizes the visual concept to be detected in the image. Next, if the concept can be found in the image, it provides a “yes” or “no” answer. Its performance matches the traditional VQA approach on unbalanced dataset, and outperforms it on the balanced dataset. <br>[contribute](contribute.md)|
|Making the V in VQA Matter|[Goyal et al.](https://arxiv.org/pdf/1612.00837.pdf)|Balances the VQA dataset by collecting complementary images such that every question is associated with a pair of similar images that result in two different answers to the question, providing a unique interpretable model that provides a counter-example based explanation.  <br>[contribute](contribute.md)|
|Visual Dialog|	[Das et al.](https://arxiv.org/abs/1611.08669)|An AI agent that holds a meaningful dialog with humans in natural, conversational language about visual content. Curates a large-scale Visual Dialog dataset (VisDial). <br>[contribute](contribute.md)|
<hr>

### Other interesting models <a name="others"/>
There are many interesting deep learning models that do not fit into the categories described above. The ONNX team would like to highly encourage users and researchers to [contribute](contribute.md) their models to the growing model zoo.

|Model Class |Reference |Description |
|-|-|-|
|Text to Image|	[Generative Adversarial Text to image Synthesis ](https://arxiv.org/abs/1605.05396)|Effectively bridges the advances in text and image modeling, translating visual concepts from characters to pixels. Generates plausible images of birds and flowers from detailed text descriptions. <br>[contribute](contribute.md)|
|Time Series Forecasting|	[Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks ](https://arxiv.org/pdf/1703.07015.pdf)|The model extracts short-term local dependency patterns among variables and to discover long-term patterns for time series trends. It helps to predict solar plant energy output, electricity consumption, and traffic jam situations. <br>[contribute](contribute.md)|
|Recommender systems|[DropoutNet: Addressing Cold Start in Recommender Systems](http://www.cs.toronto.edu/~mvolkovs/nips2017_deepcf.pdf)|A collaborative filtering method that makes predictions about an individual’s preference based on preference information from other users.<br>[contribute](contribute.md)|
|Collaborative filtering|[Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)|A DNN model based on the interaction between user and item features using matrix factorization. <br>[contribute](contribute.md)|
|Autoencoders|[A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://arxiv.org/abs/1506.01057)|An LSTM (long-short term memory) auto-encoder to preserve and reconstruct multi-sentence paragraphs.<br>[contribute](contribute.md)|
<hr>

## Usage <a name="usage-"/>

Every ONNX backend should support running the models out of the box. After downloading and extracting the tarball of each model, you will find:

- A protobuf file `model.onnx` that represents the serialized ONNX model.
- Test data (in the form of serialized protobuf TensorProto files or serialized NumPy archives).

The test data files can be used to validate ONNX models from the Model Zoo. We have provided the following interface examples for you to get started. Please replace `onnx_backend` in your code with the appropriate framework of your choice that provides ONNX inferencing support, and likewise replace `backend.run_model` with the framework's model evaluation logic. 

There are two different formats for the test data files:

- Serialized protobuf TensorProtos (.pb), stored in folders with the naming convention `test_data_set_*`.

```python
import numpy as np
import onnx
import os
import glob
import onnx_backend as backend

from onnx import numpy_helper

model = onnx.load('model.onnx')
test_data_dir = 'test_data_set_0'

# Load inputs
inputs = []
inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
for i in range(inputs_num):
    input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    inputs.append(numpy_helper.to_array(tensor))

# Load reference outputs
ref_outputs = []
ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
for i in range(ref_outputs_num):
    output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    ref_outputs.append(numpy_helper.to_array(tensor))

# Run the model on the backend
outputs = list(backend.run_model(model, inputs))

# Compare the results with reference outputs.
for ref_o, o in zip(ref_outputs, outputs):
    np.testing.assert_almost_equal(ref_o, o)
```

- Serialized Numpy archives, stored in files with the naming convention `test_data_*.npz`. Each file contains one set of test inputs and outputs.

```python
import numpy as np
import onnx
import onnx_backend as backend

# Load the model and sample inputs and outputs
model = onnx.load(model_pb_path)
sample = np.load(npz_path, encoding='bytes')
inputs = list(sample['inputs'])
outputs = list(sample['outputs'])

# Run the model with an onnx backend and verify the results
np.testing.assert_almost_equal(outputs, backend.run_model(model, inputs))
```

## Model Visualization
You can see visualizations of each model's network architecture by using [Netron](https://lutzroeder.github.io/Netron).

## Contributions
Do you want to contribute a model? To get started, pick any model presented above with the [contribute](contribute.md) link under the Description column. The links point to a page containing guidelines for making a contribution.

# License

[MIT License](LICENSE)