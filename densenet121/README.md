# DenseNet-121

Download:
- release 1.1 https://s3.amazonaws.com/download.onnx/models/opset_3/densenet121.tar.gz
- master: https://s3.amazonaws.com/download.onnx/models/opset_6/densenet121.tar.gz

Model size: 33 MB

## Description
DenseNet-121 is a convolutional neural network for classification.

### Paper
[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

### Dataset
[ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)

## Source
Caffe2 DenseNet-121 ==> ONNX DenseNet

## Model input and output
### Input
```
data_0: float[1, 3, 224, 224]
```
### Output
```
fc6_1: float[1, 1000, 1, 1]
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
