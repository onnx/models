# YOLOv2

## Description
This model is a real-time neural network for object detection that detects 20 different classes from the [VOC 2007+2012 datasets](http://host.robots.ox.ac.uk/pascal/VOC/). For information on network architecture, see the [full YOLOv2 paper](https://pjreddie.com/darknet/yolov2/). 

## Model
The model was converted to ONNX from a Core ML version of YOLOv2 using [WinMLTools](https://pypi.org/project/winmltools/). The base source code for this conversion can be found [here](https://github.com/hollance/YOLO-CoreML-MPSNNGraph). The [original network](https://pjreddie.com/darknet/yolov2/) implemented in Darknet was [modified](https://github.com/allanzelener/YAD2K/issues/80#issuecomment-347211163) to allow for conversion to Keras format. Additionally, to enable conversion to (1) Keras from Darknet format; and (2) Core ML from Keras format, modifications to layers of the `yolov2.cfg` file were required, and were performed according to [link](https://github.com/allanzelener/YAD2K/blob/master/yad2k.py).

|Model|Checksum|Download (with sample test data)| ONNX version |Opset version|
|-----|:-------|:-------------------------------|:-------------|:------------|
|YOLOv2|[MD5](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov2/yolov2-md5.txt)|[203.9 MB](https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov2/models/yolov2.onnx) |1.3  |8 |

## Inference
### Input to model
shape `(1x3x416x416)`

### Output of model
shape `(1x125x13x13)`

### Postprocessing steps
The output is a `(125x13x13)` tensor where 13x13 is the number of grid cells that the image gets divided into. Each grid cell corresponds to 5 channels, made up of the 5 bounding boxes predicted by the grid cell and the 20 classes that describe each bounding box (`5 x (20 classes + 5) = 125`). For more information on how to derive the final bounding boxes and their corresponding confidence scores, refer to this [post](http://machinethink.net/blog/object-detection-with-yolo/).

## Dataset (Train and validation)
The YOLOv2 model was trained on the [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset and was sourced from the original yolov2-voc `.cfg` and `.weights` files from [link](https://pjreddie.com/darknet/yolov2/).

## References
"YOLO9000: Better, Faster, Stronger" [arXiv:1612.08242](https://arxiv.org/pdf/1612.08242.pdf)

## License
MIT License
