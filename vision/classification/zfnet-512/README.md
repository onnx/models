8 ZFNet-512

Download:
- release 1.1: https://s3.amazonaws.com/download.onnx/models/opset_3/zfnet512.tar.gz
- release 1.1.2: https://s3.amazonaws.com/download.onnx/models/opset_6/zfnet512.tar.gz
- release 1.2: https://s3.amazonaws.com/download.onnx/models/opset_7/zfnet512.tar.gz
- release 1.3: https://s3.amazonaws.com/download.onnx/models/opset_8/zfnet512.tar.gz
- master: https://s3.amazonaws.com/download.onnx/models/opset_9/zfnet512.tar.gz

Model size: 349 MB

## Description
ZFNet-512 is a deep convolutional networks for classification.
This model's 4th layer has 512 maps instead of 1024 maps mentioned in the paper.

### Paper
[Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)

### Dataset
[ILSVRC2013](http://www.image-net.org/challenges/LSVRC/2013/)

## Source
Caffe2 ZFNet-512 ==> ONNX ZFNet-512

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
- test_data_set_0
- test_data_set_1
- test_data_set_2
- test_data_set_3
- test_data_set_4
- test_data_set_5

## Results/accuracy on test set

## License
MIT
