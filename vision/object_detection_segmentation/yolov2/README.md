# YOLOv2

## Description
This model is a real-time neural network for object detection that detects 80 different classes from the [COCO dataset](http://cocodataset.org/#home). For information on network architecture, see the [full YOLOv2 paper](https://pjreddie.com/darknet/yolov2/). 

## Model
|Model|Checksum|Download (with sample test data)| ONNX version |Opset version|
|-----|:-------|:-------------------------------|:-------------|:------------|
|YOLOv2|[MD5]()|[58 MB]() |1.3  |8 |


### Paper
"YOLO9000: Better, Faster, Stronger" [arXiv:1612.08242](https://arxiv.org/pdf/1612.08242.pdf)

### Dataset
The YOLOv2 model was trained on the [COCO](http://cocodataset.org/#home) dataset.

### Source
The model was converted to ONNX from a Core ML version of YOLOv2 using [WinMLTools](https://pypi.org/project/winmltools/). The base source code for this conversion can be found [here](https://github.com/hollance/YOLO-CoreML-MPSNNGraph). The [original network](https://pjreddie.com/darknet/yolov2/) implemented in Darknet was [modified](https://github.com/allanzelener/YAD2K/issues/80#issuecomment-347211163) to allow for conversion to Keras format. Additionally, to enable conversion to (1) Keras from Darknet format; and (2) Core ML from Keras format, modifications to layers of the `yolov2.cfg` file were required, and were performed according to [link](https://github.com/allanzelener/YAD2K/blob/master/yad2k.py).

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
