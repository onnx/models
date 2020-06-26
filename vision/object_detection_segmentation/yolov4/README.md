# YOLOv4

## Description
[YOLOv4](https://github.com/hunglc007/tensorflow-yolov4-tflite) optimizes the speed and accuracy of object detection. It is twice faster than EfficientDet and improves YOLOv3's AP and FPS by 10% and 12%, respectively. 

## Model

|Model        |Download  |Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|YOLOv4       |[251 MB](model/yolov4.onnx) |[236 MB](model/yolov4.tar.gz)|1.6 |11 |mAP of 0.5733 |



## Inference
### Conversion
A tutorial for the conversion process can be found in the [conversion](dependencies/conversion.ipynb) notebook.

Validation of the converted model and a graph representation of it can be found in the [validation](dependencies/onnx-model-validation.ipynb) notebook.
### Running inference
A tutorial for running inference using onnxruntime can be found in the [ort](dependencies/ort.ipynb) notebook.
### Input to model
Input images are resized to the shape `(1, 416, 416, 3)`.

### Preprocessing steps
The following code shows how to preprocess an image. For more information and an example on how preprocess is done, please visit [this](dependencies/ort.ipynb) notebook.

```python
import numpy as np
import cv2

# this function is from tensorflow-yolov4-tflite/core/utils.py
def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

# input
input_size = 416

original_image = cv2.imread("input.jpg")
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

```

### Output of model
Output shape: `(1, 52, 52, 3, 85)`

There are 3 output layers. For each layer, there are 255 outputs: 85 values per anchor, times 3 anchors.

The 85 values of each anchor consists of 4 box coordinates describing the predicted bounding box (x, y, h, w), 1 object confidence, and 80 class confidences. [Here](https://github.com/hunglc007/tensorflow-yolov4-tflite/blob/master/data/classes/coco.names) is the class list.

### Postprocessing steps
To see postprocess steps and an example, please visit [this](dependencies/ort.ipynb) notebook.

## Dataset
Pretrained yolov4 weights can be downloaded [here](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT). 

## Validation accuracy
mAP50 on COCO 2017 dataset is 0.5733, based on the original tensorflow [model](https://github.com/hunglc007/tensorflow-yolov4-tflite#map50-on-coco-2017-dataset).

## Publication/Attribution
* [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934). Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao. 
* Original [models](https://github.com/AlexeyAB/darknet).

## References
This model is diretly coverted from [hunglc007/tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite).

## Contributers
[Jennifer Wang](https://github.com/jennifererwangg)

## License
MIT License