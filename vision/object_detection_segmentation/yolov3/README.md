<!--- SPDX-License-Identifier: MIT -->

# YOLOv3

## Description
This model is a neural network for real-time object detection that detects 80 different classes. It is very fast and accurate.

## Model

|Model        |Download  |Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|YOLOv3       |[237 MB](model/yolov3-10.onnx) |[222 MB](model/yolov3-10.tar.gz)|1.5 |10 |mAP of 0.553 |
|YOLOv3-12    |[237 MB](model/yolov3-12.onnx) |[222 MB](model/yolov3-12.tar.gz)|1.9 |12 |mAP of 0.2874 |
|YOLOv3-12-int8 |[61 MB](model/yolov3-12-int8.onnx) |[47 MB](model/yolov3-12-int8.tar.gz)|1.9 |12 |mAP of 0.2688 |
> Compared with the YOLOv3-12, YOLOv3-12-int8's mAP decline is 0.0186 and performance improvement is 2.25x.
>
> Note the performance depends on the test hardware. 
> 
> Performance data here is collected with Intel® Xeon® Platinum 8280 Processor, 1s 4c per instance, CentOS Linux 8.3, data batch size is 1.


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
image_size = np.array([image.size[1], image.size[0]], dtype=np.int32).reshape(1, 2)
```

### Output of model
The model has 3 outputs.
boxes: `(1x'n_candidates'x4)`, the coordinates of all anchor boxes,
scores: `(1x80x'n_candidates')`, the scores of all anchor boxes per class,
indices: `('nbox'x3)`, selected indices from the boxes tensor. The selected index format is (batch_index, class_index, box_index). The class list is [here](https://github.com/qqwweee/keras-yolo3/blob/master/model_data/coco_classes.txt)

### Postprocessing steps
Post processing and meaning of output
```
out_boxes, out_scores, out_classes = [], [], []
for idx_ in indices:
    out_classes.append(idx_[1])
    out_scores.append(scores[tuple(idx_)])
    idx_1 = (idx_[0], idx_[2])
    out_boxes.append(boxes[idx_1])
```
out_boxes, out_scores, out_classes are list of resulting boxes, scores, and classes.
<hr>

## Dataset (Train and validation)
We use pretrained weights from pjreddie.com [here](https://pjreddie.com/media/files/yolov3.weights).
<hr>

## Validation accuracy
YOLOv3:
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.
mAP of 0.553 based on original Yolov3 model [here](https://pjreddie.com/darknet/yolo/)

YOLOv3-12 & YOLOv3-12-int8:
Metric is COCO box mAP@[IoU=0.50:0.95 | area=all | maxDets=100], computed over 2017 COCO val data.
<hr>

## Quantization
YOLOv3-12-int8 is obtained by quantizing YOLOv3-12 model. We use [Intel® Neural Compressor](https://github.com/intel/neural-compressor) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/object_detection/onnx_model_zoo/yolov3/quantization/ptq/README.md) to understand how to use Intel® Neural Compressor for quantization.

### Environment
onnx: 1.9.0 
onnxruntime: 1.10.0

### Prepare model
```shell
wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov3/model/yolov3-12.onnx
```

### Model quantize
```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=yolov3.yaml \
                   --data_path=path/to/COCO2017 \
                   --output_model=path/to/save
```
<hr>

## Publication/Attribution
Joseph Redmon, Ali Farhadi. YOLOv3: An Incremental Improvement, [paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

<hr>

## References
* This model is converted from a keras model [repository](https://github.com/qqwweee/keras-yolo3) using keras2onnx converter [repository](https://github.com/onnx/keras-onnx).
* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)
<hr>

## Contributors
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)
<hr>

## License
MIT License
<hr>
