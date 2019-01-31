# Open Neural Network eXchange (ONNX) Model Zoo
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](contribute.md)

<p align="center">
	<img src="images/ONNX_Model_Zoo_Graphics.png" width="60%"/>
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
* [Machine Translation](#machine_translation)
* [Language Modelling](#language)
* [Visual Question Answering & Dialog](#visual_qna)
* [Other interesting models](#others)

### Image Classification <a name="image_classification"/>
This collection of models take images as input, then classifies the major objects in the images into a set of predefined classes.

|Model Class |Reference |Description |
|-|-|-|
|<b>[MobileNet](models/image_classification/mobilenet/)</b>|[Sandler et al.](https://arxiv.org/abs/1801.04381)|Computationally efficient CNN model for mobile and embedded vision applications. <br>Top-5 error from paper - ~10%|
|<b>[ResNet](models/image_classification/resnet/)</b>|[He et al.](https://arxiv.org/abs/1512.03385), [He et al.](https://arxiv.org/abs/1603.05027)|Very deep state-of-the-art CNN model (up to 152 layers), won the ImageNet Challenge in 2015. <br>Top-5 error from  paper - ~3.6%|
|<b>[SqueezeNet](models/image_classification/squeezenet/)</b>|[Iandola et al.](https://arxiv.org/abs/1602.07360)|A light-weight CNN providing Alexnet level accuracy with 50X fewer parameters. <br>Top-5 error from  paper - ~20%|
|<b>[VGG](models/image_classification/vgg/)</b>|[Simonyan et al.](https://arxiv.org/abs/1409.1556)|Deep CNN model (up to 19 layers) which won the ImageNet Challenge in 2014. <br>Top-5 error from  paper - ~8%|
|<b>[Bvlc_AlexNet](bvlc_alexnet)</b>|[Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)|Deep CNN model for Image Classification (up to 8 layers), won the ImageNet Challenge in 2012. <br>Top-5 error from paper - ~15%|
|<b>[Bvlc_GoogleNet](bvlc_googlenet)</b>|[Szegedy et al.](https://arxiv.org/pdf/1409.4842.pdf)|Deep CNN model (up to 22 layers) implemented in Caffe and won at the ImageNet Challenge in 2014. <br>Top-5 error from paper - ~6.7%|
|<b>[Bvlc_reference_CaffeNet](bvlc_reference_caffenet)</b>|[Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)|Deep CNN variation of AlexNet for Image Classification in Caffe where the max pooling precedes the local response normalization (LRN) so that the LRN takes less compute and memory.|
|<b>[Bvlc_reference_RCNN_ILSVRC13](bvlc_reference_rcnn_ilsvrc13)</b>|[Girshick et al.](https://arxiv.org/abs/1311.2524)|Pure Caffe implementation of R-CNN for image classification as presented at CVPR in 2014.|
|<b>[DenseNet121](densenet121)</b>|[Huang et al.](https://arxiv.org/abs/1608.06993)|Deep CNN model for Image Classification, connecting every layer to every other layer.|
|<b>[Inception_v1](inception_v1)</b>|[Szegedy et al.](https://arxiv.org/abs/1409.4842)|Deep CNN model (up to 22 layers) for Image Classification - same as GoogLeNet, implemented through Caffe2. <br>Top-5 error from paper - ~6.7%|
|<b>[Inception_v2](inception_v2)</b>|[Szegedy et al.](https://arxiv.org/abs/1512.00567)|Deep CNN model for Image Classification as an adaptation to Inception v1 with batch normalization <br> Top-5 error from paper ~4.82%|
|<b>[ShuffleNet](shufflenet)</b>|[Zhang et al.](https://arxiv.org/abs/1707.01083)|Computationally efficient deep CNN model for Image Classification, providing a ~13x speedup over AlexNet on ARM-based mobile devices <br> Top-1 error from paper - ~7.8%|
|<b>[ZFNet512](zfnet512)</b>|[Zeiler et al.](https://arxiv.org/abs/1311.2901)|Deep CNN model (up to 8 layers) for Image Classification that tuned the hyperparameters of AlexNet and won the ImageNet Challenge in 2013. <br> Top-5 error from paper - ~14.3%|
<hr>

#### Domain-based Image Classification <a name="domain_based_image"/>
This subset of models classify images for specific domains and datasets.

|Model Class |Reference |Description |
|-|-|-|
|[**MNIST**- Handwritten Digit Recognition](mnist) |[Convolutional Neural Network with MNIST](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.ipynb)	|Deep CNN model for handwritten digit identification|
<hr>

### Object Detection & Image Segmentation <a name="object_detection"/>
Object detection models detect the presence of multiple objects in an image and segment out areas of the image where the objects are detected. Semantic segmentation models partition an input image by labeling each pixel into a set of pre-defined categories.

|Model Class |Reference |Description |
|-|-|-|
|<b>[Tiny_YOLOv2](tiny_yolov2)</b>|[Redmon et al.](https://arxiv.org/pdf/1612.08242.pdf)|Deep CNN model for Object Detection|
|<b>SSD</b>|[Liu et al.](https://arxiv.org/abs/1512.02325)|[contribute](contribute.md)|
|<b>Faster-RCNN</b>|[Ren et al.](https://arxiv.org/abs/1506.01497)|[contribute](contribute.md)|
|<b>Mask-RCNN</b>|[He et al.](https://arxiv.org/abs/1703.06870)|[contribute](contribute.md)|
|<b>YOLO v2</b>|[Redmon et al.](https://arxiv.org/abs/1612.08242)|[contribute](contribute.md)|
|<b>YOLO v3</b>|[Redmon et al.](https://pjreddie.com/media/files/papers/YOLOv3.pdf)|[contribute](contribute.md)|
|<b>[DUC](models/semantic_segmentation/DUC/)</b>|[Wang et al.](https://arxiv.org/abs/1702.08502)|Deep CNN based semantic segmentation model with >80% [mIOU](/models/semantic_segmentation/DUC/README.md/#metric) (mean Intersection Over Union), trained on urban street images|
|<b>FCN</b>|[Long et al.](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)|[contribute](contribute.md)|
<hr>

### Body, Face & Gesture Analysis <a name="body_analysis"/>
Face detection models identify and/or recognize human faces in images. Some more popular models are used for detection of celebrity faces, gender, age, and emotions.

|Model Class |Reference |Description |
|-|-|-|
|<b>[ArcFace](models/face_recognition/ArcFace/)</b>|[Deng et al.](https://arxiv.org/abs/1801.07698)|ArcFace is a CNN based model for face recognition which learns discriminative features of faces and produces embeddings for input face images.|
|<b>CNN Cascade</b>|[Li et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf)|[contribute](contribute.md)|
|[**Emotion FerPlus**](emotion_ferplus) |[Barsoum et al.](https://arxiv.org/abs/1608.01041)	| Deep CNN for emotion recognition trained on images of faces.|
|Age and Gender Classification using Convolutional Neural Networks| [Levi et al.](https://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf)	|[contribute](contribute.md)|
<hr>

### Image Manipulation <a name="image_manipulation"/>
Image manipulation models use neural networks to transform input images to modified output images. Some popular models in this category involve style transfer or enhancing images by increasing resolution.

|Model Class |Reference |Description |
|-|-|-|
|Unpaired Image to Image Translation using Cycle consistent Adversarial Network|[Zhu et al.](https://arxiv.org/abs/1703.10593)|[contribute](contribute.md)|
|Image Super resolution using deep convolutional networks |	[Dong et al.](http://ieeexplore.ieee.org/document/7115171/?reload=true)	|[contribute](contribute.md)|
<hr>

### Speech & Audio Processing <a name="speech"/>
This class of models uses audio data to train models that can identify voice, generate music, or even read text out loud.

|Model Class |Reference |Description |
|-|-|-|
|Speech recognition with deep recurrent neural networks|	[Graves et al.](https://www.cs.toronto.edu/~fritz/absps/RNN13.pdf)|[contribute](contribute.md)|
|Deep voice: Real time neural text to speech |	[Arik et al.](https://arxiv.org/abs/1702.07825)	|[contribute](contribute.md)|
|Sound Generative models|	[WaveNet: A Generative Model for Raw Audio ](https://arxiv.org/abs/1609.03499)|[contribute](contribute.md)|
<hr>

### Machine Translation <a name="machine_translation"/>
This class of natural language processing models learns how to translate input text to another language.

|Model Class |Reference |Description |
|-|-|-|
|Neural Machine Translation by jointly learning to align and translate|	[Bahdanau et al.](https://arxiv.org/abs/1409.0473)|[contribute](contribute.md)|
|Google's Neural Machine Translation System|	[Wu et al.](https://arxiv.org/abs/1609.08144)|[contribute](contribute.md)|
<hr>

### Language Modelling <a name="language"/>
This subset of natural language processing models learns representations of language from large corpuses of text.

|Model Class |Reference |Description |
|-|-|-|
|Deep Neural Network Language Models|	[Arisoy et al.](https://pdfs.semanticscholar.org/a177/45f1d7045636577bcd5d513620df5860e9e5.pdf)	|[contribute](contribute.md)|
<hr>

### Visual Question Answering & Dialog <a name="visual_qna"/>
This subset of natural language processing models uses input images to answer questions about those images.

|Model Class |Reference |Description |
|-|-|-|
|VQA: Visual Question Answering |[Agrawal et al.](https://arxiv.org/pdf/1505.00468v6.pdf)|[contribute](contribute.md)|
|Yin and Yang: Balancing and Answering Binary Visual Questions |[Zhang et al.](https://arxiv.org/pdf/1511.05099.pdf)|[contribute](contribute.md)|
|Making the V in VQA Matter|[Goyal et al.](https://arxiv.org/pdf/1612.00837.pdf)|[contribute](contribute.md)|
|Visual Dialog|	[Das et al.](https://arxiv.org/abs/1611.08669)|[contribute](contribute.md)|
<hr>

### Other interesting models <a name="others"/>
There are many interesting deep learning models that do not fit into the categories described above. The ONNX team would like to highly encourage users and researchers to [contribute](contribute.md) their models to the growing model zoo.

|Model Class |Reference |Description |
|-|-|-|
|Text to Image|	[Generative Adversarial Text to image Synthesis ](https://arxiv.org/abs/1605.05396)|[contribute](contribute.md)|
|Time Series Forecasting|	[Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks ](https://arxiv.org/pdf/1703.07015.pdf)|[contribute](contribute.md)|
|Recommender systems|[DropoutNet: Addressing Cold Start in Recommender Systems](http://www.cs.toronto.edu/~mvolkovs/nips2017_deepcf.pdf)|[contribute](contribute.md)|
|Collaborative filtering|[Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)|[contribute](contribute.md)|
|Autoencoders|[A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://arxiv.org/abs/1506.01057)|[contribute](contribute.md)|
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
