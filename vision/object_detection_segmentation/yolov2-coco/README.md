<!--- SPDX-License-Identifier: Apache-2.0 -->

# YOLOv2-COCO

## Description
This model aims to detect objects in real time. It detects 80 different classes from the [COCO Datasets](http://cocodataset.org/#home). For information on network architecture, see the [author's page](https://pjreddie.com/darknet/yolov2/) and [white paper](https://arxiv.org/pdf/1612.08242.pdf).

## Model
The model was converted to ONNX from PyTorch version of YOLOv2 using [PyTorch-Yolo2](https://github.com/marvis/pytorch-yolo2). The output is fully verified by generating bounding boxes under PyTorch and onnxruntime.

|Model|Download| ONNX version |Opset version|
|-----|:--------------|:-------------|:------------|
|YOLOv2|[203.9 MB](model/yolov2-coco-9.onnx) |1.5  |9 |

## Inference
### Input to model
shape `(1x3x416x416)`

### Output of model
shape `(1x425x13x13)`

### Postprocessing steps
The output is a `(1x425x13x13)` tensor where 13x13 is the number of grid cells that the image gets divided into. Each grid cell corresponds to 5 anchors, made up of the 5 bounding boxes predicted by the grid cell and the 80 classes that describe each bounding box (`5 x (80 classes + 5) = 425`). For more information on how to derive the final bounding boxes and their corresponding confidence scores, refer to this [post](https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/object-detection-onnx) and [PyTorch source code](https://github.com/marvis/pytorch-yolo2/blob/master/detect.py).

## Dataset (Train and validation)
The YOLOv2 model was trained on the [COCO](http://cocodataset.org/#home) datasets and was sourced from the original yolov2-voc `.cfg` and `.weights` files from [link](https://pjreddie.com/darknet/yolov2/).

## References
"YOLO9000: Better, Faster, Stronger" [arXiv:1612.08242](https://arxiv.org/pdf/1612.08242.pdf)

## License
Apache License v2.0
