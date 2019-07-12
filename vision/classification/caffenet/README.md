# CaffeNet
Download:
- release 1.1: https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_reference_caffenet.tar.gz
- release 1.1.2: https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_reference_caffenet.tar.gz
- release 1.2: https://s3.amazonaws.com/download.onnx/models/opset_7/bvlc_reference_caffenet.tar.gz
- release 1.3: https://s3.amazonaws.com/download.onnx/models/opset_8/bvlc_reference_caffenet.tar.gz
- master: https://s3.amazonaws.com/download.onnx/models/opset_9/bvlc_reference_caffenet.tar.gz

Model size: 244 MB

## Description
CaffeNet a variant of AlexNet.
AlexNet is the name of a convolutional neural network for classification,
which competed in the ImageNet Large Scale Visual Recognition Challenge in 2012.

Differences:
- not training with the relighting data-augmentation;
- the order of pooling and normalization layers is switched (in CaffeNet, pooling is done before normalization).

### Paper
[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

### Dataset
[ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)

## Source
Caffe BVLC CaffeNet ==> Caffe2 CaffeNet ==> ONNX CaffeNet

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
- test_data_set_0
- test_data_set_1
- test_data_set_2
- test_data_set_3
- test_data_set_4
- test_data_set_5

## Results/accuracy on test set
This model is snapshot of iteration 310,000.
The best validation performance during training was iteration
313,000 with validation accuracy 57.412% and loss 1.82328.
This model obtains a top-1 accuracy 57.4% and a top-5 accuracy
80.4% on the validation set, using just the center crop.
(Using the average of 10 crops, (4 + 1 center) * 2 mirror,
should obtain a bit higher accuracy still.)

## License
[BSD-3](LICENSE)
