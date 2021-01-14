<!--- SPDX-License-Identifier: Apache-2.0 -->

# ZFNet-512

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|
| ------------- | ------------- | ------------- | ------------- | ------------- |
|ZFNet-512| [341 MB](model/zfnet512-3.onnx)  |  [320 MB](model/zfnet512-3.tar.gz) |  1.1 | 3|
|ZFNet-512| [341 MB](model/zfnet512-6.onnx)  |  [320 MB](model/zfnet512-6.tar.gz) |  1.1.2 | 6|
|ZFNet-512| [341 MB](model/zfnet512-7.onnx)  |  [320 MB](model/zfnet512-7.tar.gz) |  1.2 | 7|
|ZFNet-512| [341 MB](model/zfnet512-8.onnx)  |  [318 MB](model/zfnet512-8.tar.gz) |  1.3 | 8|
|ZFNet-512| [341 MB](model/zfnet512-9.onnx)  |  [318 MB](model/zfnet512-9.tar.gz) |  1.4 | 9|


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
