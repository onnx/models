<!--- SPDX-License-Identifier: MIT -->

# MNIST - Handwritten Digit Recognition

## Description
This model predicts handwritten digits using a convolutional neural network (CNN).

## Model
|Model|Download|Download (with sample test data)| ONNX version |Opset version|TOP-1 ERROR|
|-----|:-------|:-------------------------------|:-------------|:------------|:------------|
|MNIST|[27 kB](model/mnist-1.onnx)|[26 kB](model/mnist-1.tar.gz) |1.0  |1 |1.1% |
|MNIST|[26 kB](model/mnist-7.onnx)|[26 kB](model/mnist-7.tar.gz) |1.2  |7 |1.1% |
|MNIST|[26 kB](model/mnist-8.onnx)|[26 kB](model/mnist-8.tar.gz) |1.3  |8 |1.1% |
|MNIST-12|[26 kB](model/mnist-12.onnx)|[26 kB](model/mnist-12.tar.gz) |1.9  |12 |1.1% |
|MNIST-12-int8|[11 kB](model/mnist-12-int8.onnx)|[10 kB](model/mnist-12-int8.tar.gz) |1.9  |12 |1.1% |

### Dataset
The model has been trained on the popular [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

### Source
The model is trained in CNTK following the tutorial [CNTK 103D: Convolutional Neural Network with MNIST](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.ipynb). Note that the specific architecture used is the model with alternating convolution and max pooling layers (found under the "Solution" section at the end of the tutorial).

### Demo
[Run MNIST in browser](https://microsoft.github.io/onnxjs-demo/#/mnist) - implemented by ONNX.js with MNIST version 1.2

## Inference
We used CNTK as the framework to perform inference. A brief description of the inference process is provided below:

### Input
Input tensor has shape `(1x1x28x28)`, with type of float32.
One image at a time. This model doesn't support mini-batch.

### Preprocessing
Images are resized into (28x28) in grayscale, with a black background and a white foreground (the number should be in white). Color value is scaled to [0.0, 1.0].

Example:
```python
import numpy as np
import cv2

image = cv2.imread('input.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (28,28)).astype(np.float32)/255
input = np.reshape(gray, (1,1,28,28)
```

### Output
The likelihood of each number before [softmax](https://en.wikipedia.org/wiki/Softmax_function), with shape of `(1x10)`.

### Postprocessing
Route the model output through a softmax function to map the aggregated activations across the network to probabilities across the 10 classes.

### Sample test data
Sets of sample input and output files are provided in
* serialized protobuf TensorProtos (`.pb`), which are stored in the folders `test_data_set_*/`.

## Quantization
MNIST-12-int8 is obtained by quantizing MNIST-12 model. We use [Intel® Neural Compressor](https://github.com/intel/neural-compressor) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/image_recognition/onnx_model_zoo/mnist/quantization/ptq/README.md) to understand how to use Intel® Neural Compressor for quantization.

### Environment
onnx: 1.9.0 
onnxruntime: 1.10.0

### Prepare model
```shell
wget https://github.com/onnx/models/raw/main/vision/classification/mnist/model/mnist-12.onnx
```

### Model quantize
```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=mnist.yaml \
                   --output_model=path/to/save
```

## References
* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)

## Contributors
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)

## License
MIT
