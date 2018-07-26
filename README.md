# Open Neural Network eXchange (ONNX) Model Zoo

[![Generic badge](https://img.shields.io/badge/Status-Work_In_Progress-red.svg)](#) 
[![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](contribute.md)

<p align="center">
	<img src="images/ONNX_Model_Zoo_Graphics.png" width="60%"/>
</p>

The ONNX Model Zoo is a collection of pre-trained models for state-of-the-art models in deep learning, available in the ONNX format. Accompanying each model are [Jupyter](http://jupyter.org) notebooks for model training and running inference with the trained model. The notebooks are written in Python and include links to the training dataset as well as references to the original paper that describes the model architecture. The notebooks can be exported and run as python(.py) files.

## What is ONNX?
The Open Neural Network eXchange ([ONNX](http://onnx.ai)) is a open format to represent deep learning models. With ONNX, developers can move models between state-of-the-art tools and choose the combination that is best for them. ONNX is developed and supported by a community of partners.

## Models

### Image Classification
This collection of models take images as input, then classifies the major objects in the images into a set of predefined classes.

|Model Class |Reference |Description |
|-|-|-|
|<b>[MobileNet](models/image_classification/mobilenet/)</b>|[Sandler et al.](https://arxiv.org/abs/1801.04381)|Efficient CNN model for mobile and embedded vision applications. <br>Top-5 error from paper - ~10%|
|<b>[ResNet](models/image_classification/resnet/)</b>|[He et al.](https://arxiv.org/abs/1512.03385), [He et al.](https://arxiv.org/abs/1603.05027)|Very deep CNN model (up to 152 layers), won the ImageNet Challenge in 2015. <br>Top-5 error from  paper - ~6%|
|<b>[SqueezeNet](models/image_classification/squeezenet/)</b>|[Iandola et al.](https://arxiv.org/abs/1602.07360)|A light-weight CNN providing Alexnet level accuracy with 50X fewer parameters. <br>Top-5 error from  paper - ~20%|
|<b>[VGG](models/image_classification/vgg/)</b>|[Simonyan et al.](https://arxiv.org/abs/1409.1556)|Deep CNN model (upto 19 layers) which won the ImageNet Challenge in 2014. <br>Top-5 error from  paper - ~8%|
|<b>[Bvlc_AlexNet](bvlc_alexnet)</b>|[Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)|Deep CNN model for Image Classification |
|<b>[Bvlc_GoogleNet](bvlc_googlenet)</b>|[Szegedy et al.](https://arxiv.org/pdf/1409.4842.pdf)|Deep CNN model for Image Classification|
|<b>[Bvlc_reference_CaffeNet](bvlc_reference_caffenet)</b>|[Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)|Deep CNN model for Image Classification|
|<b>[Bvlc_reference_RCNN_ILSVRC13](bvlc_reference_rcnn_ilsvrc13)</b>|[Girshick et al.](https://arxiv.org/abs/1311.2524)|Deep CNN model for Image Classification|
|<b>[DenseNet121](densenet121)</b>|[Huang et al.](https://arxiv.org/abs/1608.06993)|Deep CNN model for Image Classification|
|<b>[Inception_v1](inception_v1)</b>|[Szegedy et al.](https://arxiv.org/abs/1409.4842)|Deep CNN model for Image Classification|
|<b>[Inception_v2](inception_v2)</b>|[Szegedy et al.](https://arxiv.org/abs/1512.00567)|Deep CNN model for Image Classification|
|<b>[ShuffleNet](shufflenet)</b>|[Zhang et al.](https://arxiv.org/abs/1707.01083)|Deep CNN model for Image Classification|
|<b>[ZFNet512](zfnet512)</b>|[Zeiler et al.](https://arxiv.org/abs/1311.2901)|Deep CNN model for Image Classification|

<hr>

### Face Detection and Recognition
These models detect and/or recognize human faces in images. Some more popular models are used for detection/recognition of celebrity faces, gender, age, and emotions.

|Model Class |Reference |Description |
|-|-|-|
|<b>[ArcFace](models/face_recognition/ArcFace/)</b>|[Deng et al.](https://arxiv.org/abs/1801.07698)|ArcFace is a CNN based model for face recognition which learns discriminative features of faces and produces embeddings for input face images.|
|<b>CNN Cascade</b>|[Li et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf)|[contribute](contribute.md)|

<hr>

### Semantic Segmentation
Semantic segmentation models partition an input image by labeling each pixel into a set of pre-defined categories.

|Model Class |Reference |Description |
|-|-|-|
|<b>[DUC](models/semantic_segmentation/DUC/)</b>|[Wang et al.](https://arxiv.org/abs/1702.08502)|Deep CNN based model with >80% [mIOU](/models/semantic_segmentation/DUC/README.md/#metric) (mean Intersection Over Union) trained on urban street images|
|<b>FCN</b>|[Long et al.](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)|[contribute](contribute.md)|

<hr>

### Object Detection & Segmentation
These models detect the presence of multiple objects in an image and segment out areas of the image where the objects are detected.

|Model Class |Reference |Description |
|-|-|-|
|<b>[Tiny_YOLOv2](tiny_yolov2)</b>|[Redmon et al.](https://arxiv.org/pdf/1612.08242.pdf)|Deep CNN model for Object Detection|
|<b>SSD</b>|[Liu et al.](https://arxiv.org/abs/1512.02325)|[contribute](contribute.md)|
|<b>Faster-RCNN</b>|[Ren et al.](https://arxiv.org/abs/1506.01497)|[contribute](contribute.md)|
|<b>Mask-RCNN</b>|[He et al.](https://arxiv.org/abs/1703.06870)|[contribute](contribute.md)|
|<b>YOLO v2</b>|[Redmon et al.](https://arxiv.org/abs/1612.08242)|[contribute](contribute.md)|
|<b>YOLO v3</b>|[Redmon et al.](https://pjreddie.com/media/files/papers/YOLOv3.pdf)|[contribute](contribute.md)|

<hr>

### Emotion Recognition

|Model Class |Reference |Description |
|-|-|-|
|[Emotion FerPlus](emotion_ferplus) |[Barsoum et al.](https://arxiv.org/abs/1608.01041)	|Deep CNN model for Emotion recognition|
<hr>

### Hand Written Digit Recognition

|Model Class |Reference |Description |
|-|-|-|
|[MNIST- Hand Written Digit Recognition](mnist) |[Convolutional Neural Network with MNIST](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.ipynb)	|Deep CNN model for hand written digit identification|
<hr>

### Super Resolution

|Model Class |Reference |Description |
|-|-|-|
|Image Super resolution using deep convolutional networks |	[Dong et al.](http://ieeexplore.ieee.org/document/7115171/?reload=true)	|[contribute](contribute.md)|
<hr>

### Gender Detection

|Model Class |Reference |Description |
|-|-|-|
|Age and Gender Classification using Convolutional Neural Networks| [Levi et al.](https://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf)	|[contribute](contribute.md)|
<hr>

### Style Transfer
|Model Class |Reference |Description |
|-|-|-|
|Unpaired Image to Image Translation using Cycle consistent Adversarial Network|[Zhu et al.](https://arxiv.org/abs/1703.10593)|[contribute](contribute.md)|
<hr>

### Machine Translation

|Model Class |Reference |Description |
|-|-|-|
|Neural Machine Translation by jointly learning to align and translate|	[Bahdanau et al.](https://arxiv.org/abs/1409.0473)|[contribute](contribute.md)|
|Google's Neural Machine Translation System|	[Wu et al.](https://arxiv.org/abs/1609.08144)|[contribute](contribute.md)|
<hr>

### Speech Processing

|Model Class |Reference |Description |
|-|-|-|
|Speech recognition with deep recurrent neural networks|	[Graves et al.](https://www.cs.toronto.edu/~fritz/absps/RNN13.pdf)|[contribute](contribute.md)|
|Deep voice: Real time neural text to speech |	[Arik et al.](https://arxiv.org/abs/1702.07825)	|[contribute](contribute.md)|

<hr>

### Language Modelling

|Model Class |Reference |Description |
|-|-|-|
|Deep Neural Network Language Models|	[Arisoy et al.](https://pdfs.semanticscholar.org/a177/45f1d7045636577bcd5d513620df5860e9e5.pdf)	|[contribute](contribute.md)|
<hr>

### Visual Question Answering & Dialog

|Model Class |Reference |Description |
|-|-|-|
|VQA: Visual Question Answering |[Agrawal et al.](https://arxiv.org/pdf/1505.00468v6.pdf)|[contribute](contribute.md)|
|Yin and Yang: Balancing and Answering Binary Visual Questions |[Zhang et al.](https://arxiv.org/pdf/1511.05099.pdf)|[contribute](contribute.md)|
|Making the V in VQA Matter|[Goyal et al.](https://arxiv.org/pdf/1612.00837.pdf)|[contribute](contribute.md)|
|Visual Dialog|	[Das et al.](https://arxiv.org/abs/1611.08669)|[contribute](contribute.md)|

<hr>

### Other interesting models

|Model Class |Reference |Description |
|-|-|-|
|Text to Image|	[Generative Adversarial Text to image Synthesis ](https://arxiv.org/abs/1605.05396)|[contribute](contribute.md)|
|Sound Generative models|	[WaveNet: A Generative Model for Raw Audio ](https://arxiv.org/abs/1609.03499)|[contribute](contribute.md)|
|Time Series Forecasting|	[Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks ](https://arxiv.org/pdf/1703.07015.pdf)|[contribute](contribute.md)|
|Recommender systems|[DropoutNet: Addressing Cold Start in Recommender Systems](http://www.cs.toronto.edu/~mvolkovs/nips2017_deepcf.pdf)|[contribute](contribute.md)|
|Collaborative filtering||[contribute](contribute.md)|
|Autoencoders||[contribute](contribute.md)|

<hr>

## Model Visualization
You can see visualizations of each model's network architecture by using [Netron](https://lutzroeder.github.io/Netron).

## Usage

Every ONNX backend should support running these models out of the box. After downloading and extracting the tarball of each model, there should be

- A protobuf file `model.onnx` which is the serialized ONNX model.
- Test data.


The test data are provided in two different formats:
- Serialized Numpy archives, which are files named like `test_data_*.npz`, each file contains one set of test inputs and outputs.
They can be used like this:

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

- Serialized protobuf TensorProtos, which are stored in folders named like `test_data_set_*`.
They can be used as the following:
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
## Contributions
Do you want to contribute a model? To get started, pick any model presented above with the [contribute](contribute.md) link under the Description column. The links point to a page containing guidelines for making a contribution.

# License

[MIT License](LICENSE)
