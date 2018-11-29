# MNIST - Handwritten Digit Recognition

## Description
This model predicts handwritten digits using a convolutional neural network (CNN). 

## Model
|Model|Checksum|Download (with sample test data)| ONNX version |Opset version|
|-----|:-------|:-------------------------------|:-------------|:------------|
|MNIST|[MD5](https://onnxzoo.blob.core.windows.net/models/opset_1/mnist/mnist-md5.txt)|[26 kB](https://onnxzoo.blob.core.windows.net/models/opset_1/mnist/mnist.tar.gz) |1.0  |1 |
|     |[MD5](https://onnxzoo.blob.core.windows.net/models/opset_7/mnist/mnist-md5.txt)|[26 kB](https://onnxzoo.blob.core.windows.net/models/opset_7/mnist/mnist.tar.gz) |1.2  |7 |
|     |[MD5](https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist-md5.txt)|[26 kB](https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz) |1.3  |8 |

### Dataset
The model has been trained on the popular [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

### Source
The model is trained in CNTK following the tutorial [CNTK 103D: Convolutional Neural Network with MNIST](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.ipynb). Note that the specific architecture used is the model with alternating convolution and max pooling layers (found under the "Solution" section at the end of the tutorial).

### Demo
[Demo implemented by ONNX.js (model file: MNIST V1.2)](https://microsoft.github.io/onnxjs-demo/#/mnist)

## Inference
We used CNTK as the framework to perform inference. A brief description of the inference process is provided below:

### Input
shape `(1x1x28x28)`

### Preprocessing

### Output
shape `(1x10)`

### Postprocessing
Route the model output through a softmax function to map the aggregated activations across the network to probabilities across the 10 classes.

### Sample test data
Sets of sample input and output files are provided in 
* serialized protobuf TensorProtos (`.pb`), which are stored in the folders `test_data_set_*/`.

## License
MIT
