# MNIST - Handwritten Digit Recognition

Download package:  
- release 1.0: https://www.cntk.ai/OnnxModels/mnist/opset_1/mnist.tar.gz
- release 1.2: https://www.cntk.ai/OnnxModels/mnist/opset_7/mnist.tar.gz

Model size: 26 kB

## Description
This model predicts handwritten digits using a convolutional neural network (CNN). 

### Dataset
The model has been trained on the popular [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Source
The model is trained in CNTK following the tutorial [CNTK 103D: Convolutional Neural Network with MNIST](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.ipynb). Note that the specific architecture used is the model with alternating convolution and max pooling layers (found under the "Solution" section at the end of the tutorial).

## Model input and output
### Input
shape `(1x1x28x28)`
### Output
shape `(1x10)`

### Pre-processing steps

### Post-processing steps
Route the model output through a softmax function to map the aggregated activations across the network to probabilities across the 10 classes.

### Sample test data
Sets of sample input and output files are provided in 
* the.npz format (`test_data_*.npz`). The input is a `(1x1x28x28)` numpy array of an MNIST test image, while the output is an array of length 10 corresponding to the output of evaluating the model on the sample input.
* serialized protobuf TensorProtos (`.pb`), which are stored in the folders `test_data_set_*/`.

## License
MIT
