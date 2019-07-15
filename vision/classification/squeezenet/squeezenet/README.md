# SqueezeNet

Download:
- release 1.1: https://s3.amazonaws.com/download.onnx/models/opset_3/squeezenet.tar.gz
- release 1.1.2: https://s3.amazonaws.com/download.onnx/models/opset_6/squeezenet.tar.gz
- release 1.2: https://s3.amazonaws.com/download.onnx/models/opset_7/squeezenet.tar.gz
- release 1.3: https://s3.amazonaws.com/download.onnx/models/opset_8/squeezenet.tar.gz
- master: https://s3.amazonaws.com/download.onnx/models/opset_9/squeezenet.tar.gz

Model size: 5 MB

## Description
SqueezeNet is a light-weight convolutional networks for classification.

### Paper
[SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

### Dataset
[ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)

## Source
Caffe2 SqueezeNet ==> ONNX SqueezeNet

### Demo
[Run SqueezeNet in browser](https://microsoft.github.io/onnxjs-demo/#/squeezenet) - implemented by ONNX.js with SqueezeNet release 1.2

## Model input and output
### Input
```
data_0: float[1, 3, 224, 224]
```
### Output
```
softmaxout_1: float[1, 1000, 1, 1]
```
### Pre-processing steps
### Post-processing steps
### Sample test data
random generated sampe test data:
- test_data_set_0
- test_data_set_1
- test_data_set_2

## Results/accuracy on test set

## License
MIT
