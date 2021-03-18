<!--- SPDX-License-Identifier: MIT -->

# Tiny YOLOv2

## Description
This model is a real-time neural network for object detection that detects 20 different classes. It is made up of 9 convolutional layers and 6 max-pooling layers and is a smaller version of the more complex full [YOLOv2](https://pjreddie.com/darknet/yolov2/) network.

CoreML TinyYoloV2 ==> ONNX TinyYoloV2

## Model
|Model|Download|Download (with sample test data)| ONNX version |Opset version|
|-----|:-------|:-------------------------------|:-------------|:------------|
|Tiny YOLOv2|[62 MB](model/tinyyolov2-7.onnx)|[59 MB](model/tinyyolov2-7.tar.gz) |1.2  |7 |
|     |[62 MB](model/tinyyolov2-8.onnx)|[59 MB](model/tinyyolov2-8.tar.gz) |1.3  |8 |

### Paper
"YOLO9000: Better, Faster, Stronger" [arXiv:1612.08242](https://arxiv.org/pdf/1612.08242.pdf)

### Dataset
The Tiny YOLO model was trained on the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset.

### Source
The model was converted from a Core ML version of Tiny YOLO using [ONNXMLTools](https://github.com/onnx/onnxmltools). The source code can be found [here](https://github.com/hollance/YOLO-CoreML-MPSNNGraph). The Core ML model in turn was converted from the [original network](https://pjreddie.com/darknet/yolov2/) implemented in Darknet (via intermediate conversion through Keras).

## Inference
### Input
shape `(1x3x416x416)`
### Preprocessing
### Output
shape `(1x125x13x13)`
### Postprocessing
The output is a `(125x13x13)` tensor where 13x13 is the number of grid cells that the image gets divided into. Each grid cell corresponds to 125 channels, made up of the 5 bounding boxes predicted by the grid cell and the 25 data elements that describe each bounding box (`5x25=125`). For more information on how to derive the final bounding boxes and their corresponding confidence scores, refer to this [post](http://machinethink.net/blog/object-detection-with-yolo/).
### Sample test data
Sets of sample input and output files are provided in
* serialized protobuf TensorProtos (`.pb`), which are stored in the folders `test_data_set_*/`.

## License
MIT
