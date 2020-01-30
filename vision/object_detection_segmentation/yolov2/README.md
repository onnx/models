# YOLOv2

## Description
This model is a real-time neural network for object detection that detects 80 different classes from the [COCO dataset](http://cocodataset.org/#home). For information on network architecture, see the [full YOLOv2 paper](https://pjreddie.com/darknet/yolov2/). 

## Model
|Model|Checksum|Download (with sample test data)| ONNX version |Opset version|
|-----|:-------|:-------------------------------|:-------------|:------------|
|YOLOv2|[MD5](https://onnxzoo.blob.core.windows.net/models/opset_1/yolov2/yolov2-md5.txt)|[58 MB](https://onnxzoo.blob.core.windows.net/models/opset_1/yolov2/yolov2.tar.gz) |1.3  |8 |


### Paper
"YOLO9000: Better, Faster, Stronger" [arXiv:1612.08242](https://arxiv.org/pdf/1612.08242.pdf)

### Dataset
The YOLOv2 model was trained on the [COCO](http://cocodataset.org/#home) dataset.

### Source
The model was converted from a Core ML version of YOLOv2 using [WinMLTools](https://pypi.org/project/winmltools/). The source code can be found [here](https://github.com/hollance/YOLO-CoreML-MPSNNGraph). The Core ML model in turn was converted from the [original network](https://pjreddie.com/darknet/yolov2/) implemented in Darknet (via intermediate conversion through Keras) after performing modifications to fix the `merge` operator in the `route` layers of the `yolov2.cfg` file according to [the following link](https://github.com/allanzelener/YAD2K/issues/80#issuecomment-347211163).

## Inference
### Input
shape `(1x3x416x416)`
### Preprocessing
### Output
shape `(1x425x13x13)`
### Postprocessing
The output is a `(425x13x13)` tensor where 13x13 is the number of grid cells that the image gets divided into. Each grid cell corresponds to 425 channels, made up of the 5 bounding boxes predicted by the grid cell and the 25 data elements that describe each bounding box (`5x85=425`). For more information on how to derive the final bounding boxes and their corresponding confidence scores, refer to this [post](http://machinethink.net/blog/object-detection-with-yolo/).

## License
MIT
