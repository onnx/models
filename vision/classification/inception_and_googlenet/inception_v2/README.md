<!--- SPDX-License-Identifier: MIT -->

# Inception v2

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|
| ------------- | ------------- | ------------- | ------------- | ------------- |
|Inception-2| [44 MB](model/inception-v2-3.onnx)  |  [44 MB](model/inception-v2-3.tar.gz) |  1.1 | 3|
|Inception-2| [44 MB](model/inception-v2-6.onnx)  |  [44 MB](model/inception-v2-6.tar.gz) |  1.1.2 | 6|
|Inception-2| [44 MB](model/inception-v2-7.onnx)  |  [44 MB](model/inception-v2-7.tar.gz) |  1.2 | 7|
|Inception-2| [44 MB](model/inception-v2-8.onnx)  |  [44 MB](model/inception-v2-8.tar.gz) |  1.3 | 8|
|Inception-2| [44 MB](model/inception-v2-9.onnx)  |  [44 MB](model/inception-v2-9.tar.gz) |  1.4 | 9|

## Description
Inception v2 is a deep convolutional networks for classification.

### Paper
[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

### Dataset
[ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)

## Source
Caffe2 Inception v2 ==> ONNX Inception v2

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
