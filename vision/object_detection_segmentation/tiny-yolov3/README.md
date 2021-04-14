<!--- SPDX-License-Identifier: MIT -->

# Tiny YOLOv3

## Description
This model is a neural network for real-time object detection that detects 80 different classes. It is very fast and accurate. It is a smaller version of YOLOv3 model.

## Model

|Model        |Download  |Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|Tiny YOLOv3       |[34 MB](model/tiny-yolov3-11.onnx) |[33 MB](model/tiny-yolov3-11.tar.gz)|1.6 |11 |mAP of 0.331 |



<hr>

## Inference

### Input to model
Resized image `(1x3x416x416)`
Original image size `(1x2)` which is `[image.size[1], image.size[0]]`

### Preprocessing steps
The images have to be loaded in to a range of [0, 1]. The transformation should preferrably happen at preprocessing.

The following code shows how to preprocess a NCHW tensor:

```python
import numpy as np
from PIL import Image

# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

image = Image.open(img_path)
# input
image_data = preprocess(image)
image_size = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)
```

### Output of model
The model has 3 outputs.
boxes: `(1x'n_candidates'x4)`, the coordinates of all anchor boxes,
scores: `(1x80x'n_candidates')`, the scores of all anchor boxes per class,
indices: `('nbox'x3)`, selected indices from the boxes tensor. The selected index format is (batch_index, class_index, box_index). The class list is [here](https://github.com/qqwweee/keras-yolo3/blob/master/model_data/coco_classes.txt)

### Postprocessing steps
Post processing and meaning of output
```python
out_boxes, out_scores, out_classes = [], [], []
for idx_ in indices[0]:
    out_classes.append(idx_[1])
    out_scores.append(scores[tuple(idx_)])
    idx_1 = (idx_[0], idx_[2])
    out_boxes.append(boxes[idx_1])
```
out_boxes, out_scores, out_classes are list of resulting boxes, scores, and classes.
<hr>

## Dataset (Train and validation)
We use pretrained weights from pjreddie.com [here](https://pjreddie.com/media/files/yolov3-tiny.weights).
<hr>

## Validation accuracy
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.
mAP of 0.331 based on original tiny Yolov3 model [here](https://pjreddie.com/darknet/yolo/)
<hr>

## Publication/Attribution
Joseph Redmon, Ali Farhadi. YOLOv3: An Incremental Improvement, [paper](https://arxiv.org/pdf/1804.02767.pdf)

<hr>

## References
This model is converted from a keras model [repository](https://github.com/qqwweee/keras-yolo3) using keras2onnx converter [repository](https://github.com/onnx/keras-onnx).
<hr>

## License
MIT License
<hr>
