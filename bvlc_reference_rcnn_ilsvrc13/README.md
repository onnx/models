# R-CNN ILSVRC13

Download:
- release 1.1: https://s3.amazonaws.com/download.onnx/models/opset_3/bvlc_reference_rcnn_ilsvrc13.tar.gz
- release 1.1.2: https://s3.amazonaws.com/download.onnx/models/opset_6/bvlc_reference_rcnn_ilsvrc13.tar.gz
- release 1.2 and master: https://s3.amazonaws.com/download.onnx/models/opset_7/bvlc_reference_rcnn_ilsvrc13.tar.gz

Model size: 231 MB

## Description
R-CNN is a convolutional neural network for detection.
This model was made by transplanting the R-CNN SVM classifiers into a fc-rcnn classification layer.

### Paper
[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)

### Dataset
[ILSVRC2013](http://www.image-net.org/challenges/LSVRC/2013/)

## Source
Caffe BVLC R-CNN ILSVRC13 ==> Caffe2 R-CNN ILSVRC13 ==> ONNX R-CNN ILSVRC13

## Model input and output
### Input
```
data_0: float[1, 3, 224, 224]
```
### Output
```
fc-rcnn_1: float[1, 200]
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
On the 200-class ILSVRC2013 detection dataset, R-CNN’s mAP is 31.4%.

## License
[BSD-3](LICENSE)
