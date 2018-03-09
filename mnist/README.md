# MNIST - Handwritten Digit Recognition

Download package: https://www.cntk.ai/OnnxModels/mnist.tar.gz

## Description
This model predicts handwritten digits using a convolutional neural network (CNN). It has been trained on the popular [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Source
The model is trained in CNTK following the tutorial [CNTK 103D: Convolutional Neural Network with MNIST](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.ipynb). Note that the specific architecture used is the model with alternating convolution and max pooling layers (found under the "Solution" section at the end of the tutorial).

## Model input and output
The model expects an input image of the shape (1x28X28), scaled to between `[0,1]` (computed by: `image/255`).

Sets of sample input and output files are provided in .npz format (`test_data_*.npz`). The input is a (1x28x28) numpy array of a test image, while the output is an array of length 10 corresponding to the output of evaluating the model on the sample input.

## License
MIT