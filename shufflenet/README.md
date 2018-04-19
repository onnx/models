# ShuffleNet

Download:
- release 1.1 https://s3.amazonaws.com/download.onnx/models/opset_3/shufflenet.tar.gz
- master: https://s3.amazonaws.com/download.onnx/models/opset_6/shufflenet.tar.gz

Model size: 5.3 MB

## Description
ShuffleNet is a deep convolutional networks for classification.

### Paper
[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)

### Dataset
[ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)

## Source
Caffe2 ShuffleNet ==> ONNX ShuffleNet

## Model input and output
### Input
```
gpu_0/data_0: float[1, 3, 224, 224]
```
### Output
```
gpu_0/softmax_1: float[1, 1000]
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
