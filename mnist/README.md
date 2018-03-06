# MNIST - hand-written digit recognition

Download package: https://www.cntk.ai/OnnxModels/mnist.tar.gz

### Description
This model predicts handwritten digits using a convolutional neural network (CNN). It has been trained on the popular [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

### Tutorial
The model is trained in CNTK following the tutorial [CNTK 103D: Convolutional Neural Network with MNIST](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.ipynb). Note that the specific architecture used is the model with alternating convolution and max pooling layers (found under the "Solution" section at the end of the tutorial).

### Sample data
Sets of sample input and output files are provided in .npz format (`test_data_*.npz`).The input image of 28x28 pixels is flattened to an array of length 784, while the output (label) is one-hot encoded as an array of length 10 (corresponding to the 10 digits).

### License
MIT
