# Inception v1

Download:
- release 1.1: https://s3.amazonaws.com/download.onnx/models/opset_3/inception_v1.tar.gz
- release 1.1.2: https://s3.amazonaws.com/download.onnx/models/opset_6/inception_v1.tar.gz
- release 1.2: https://s3.amazonaws.com/download.onnx/models/opset_7/inception_v1.tar.gz
- master: https://s3.amazonaws.com/download.onnx/models/opset_8/inception_v1.tar.gz

Model size: 28 MB

## Description
Inception v1 is a reproduction of GoogLeNet.

### Paper
[Going deeper with convolutions](https://arxiv.org/abs/1409.4842)

### Dataset
[ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)

## Source
Caffe2 Inception v1 ==> ONNX Inception v1

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
