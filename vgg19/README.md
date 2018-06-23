# VGG-19

Download:
- release 1.1: https://s3.amazonaws.com/download.onnx/models/opset_3/vgg19.tar.gz
- release 1.1.2: https://s3.amazonaws.com/download.onnx/models/opset_6/vgg19.tar.gz
- release 1.2 and master: https://s3.amazonaws.com/download.onnx/models/opset_7/vgg19.tar.gz

Model size: 575 MB

## Description
VGG-19 is a deep convolutional networks for classification.

### Paper
[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

### Dataset
[ILSVRC2014](http://www.image-net.org/challenges/LSVRC/2014/)

## Source
Caffe2 VGG-19 ==> ONNX VGG-19

## Model input and output
### Input
```
data_0: float[1, 3, 224, 224]
```
### Output
```
prob_1: float[1, 1000]
```
### Pre-processing steps
### Post-processing steps
### Sample test data
random generated sampe test data:
- test_data_0.npz
- test_data_1.npz
- test_data_2.npz
- test_data_set_0
- test_data_set_1
- test_data_set_2

## Results/accuracy on test set

## License
MIT
