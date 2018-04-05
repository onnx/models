# Tiny YOLOv2

Download: https://www.cntk.ai/OnnxModels/tiny_yolov2.tar.gz  
Model size: 61 MB

## Description
This model is a real-time neural network for object detection that detects 20 different classes. It is made up of 9 convolutional layers and 6 max-pooling layers and is a smaller version of the more complex full [YOLOv2](https://pjreddie.com/darknet/yolov2/) network. 

### Paper
"YOLO9000: Better, Faster, Stronger" [arXiv:1612.08242](https://arxiv.org/pdf/1612.08242.pdf)

### Dataset
The Tiny YOLO model was trained on the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset.

## Source
The model was converted from a Core ML version of Tiny YOLO using [ONNXMLTools](https://github.com/onnx/onnxmltools). The source code can be found [here](https://github.com/hollance/YOLO-CoreML-MPSNNGraph). The Core ML model in turn was converted from the [original network](https://pjreddie.com/darknet/yolov2/) implemented in Darknet (via intermediate conversion to Keras).

## Model input and output
### Input
Input image of the shape (3x416x416)
### Output
Output is a (1x125x13x13) array
### Pre-processing steps
Resize the input image to a (3x416x416) array of type float32.
### Post-processing steps
The output is a (125x13x13) tensor where 13x13 is the number of grid cells that the image gets divided into. Each grid cell corresponds to 125 channels, made up of the 5 bounding boxes predicted by the grid cell and the 25 data elements that describe each bounding box (`5x25=125`). For more information on how to derive the final bounding boxes and their corresponding confidence scores, refer to this [post](http://machinethink.net/blog/object-detection-with-yolo/).
### Sample test data
Sets of sample input and output files are provided in .npz format (`test_data_*.npz`). The input is a (3x416x416) numpy array of a test image from Pascal VOC, while the output is a numpy array of shape (1x125x13x13).

## License
MIT
