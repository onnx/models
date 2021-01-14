<!--- SPDX-License-Identifier: Apache-2.0 -->

# Mask R-CNN

## Description
This model is a real-time neural network for object instance segmentation that detects 80 different [classes](dependencies/coco_classes.txt).

## Model

|Model        |Download  | Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|Mask R-CNN R-50-FPN      |[177.9 MB](model/MaskRCNN-10.onnx) | [168.8 MB](model/MaskRCNN-10.tar.gz) |1.5 |10 |mAP of 0.36 & 0.33 |


<hr>

## Inference

### Input to model
Image shape `(3x'height'x'width')`

### Preprocessing steps
The images have to be loaded in to a range of [0, 255], resized, converted to BGR and then normalized using mean = [102.9801, 115.9465, 122.7717]. The transformation should preferably happen at preprocessing.

This model can take images of different sizes as input. However, to achieve best performance, it is recommended to resize the image such that both height and width are within the range of [800, 1333], and then pad the image with zeros such that both height and width are divisible by 32.

The following code shows how to preprocess the [demo image](dependencies/demo.jpg):

```python
import numpy as np
from PIL import Image

def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image

img = Image.open('dependencies/demo.jpg')
img_data = preprocess(img)
```

### Output of model
The model has 4 outputs.

boxes: `('nbox'x4)`, in `(xmin, ymin, xmax, ymax)`.

labels: `('nbox')`.

scores: `('nbox')`.

masks: `('nbox', 1, 28, 28)`.

### Postprocessing steps

The following code shows how to patch the original image with detections, class annotations and segmentation, filtered by scores:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pycocotools.mask as mask_util
import cv2

classes = [line.rstrip('\n') for line in open('coco_classes.txt')]

def display_objdetect_image(image, boxes, labels, scores, masks, score_threshold=0.7):
    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes /= ratio

    _, ax = plt.subplots(1, figsize=(12,9))

    image = np.array(image)

    for mask, box, label, score in zip(masks, boxes, labels, scores):
        # Showing boxes with score > 0.7
        if score <= score_threshold:
            continue

        # Finding contour based on mask
        mask = mask[0, :, :, None]
        int_box = [int(i) for i in box]
        mask = cv2.resize(mask, (int_box[2]-int_box[0]+1, int_box[3]-int_box[1]+1))
        mask = mask > 0.5
        im_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        x_0 = max(int_box[0], 0)
        x_1 = min(int_box[2] + 1, image.shape[1])
        y_0 = max(int_box[1], 0)
        y_1 = min(int_box[3] + 1, image.shape[0])
        mask_y_0 = max(y_0 - box[1], 0)
        mask_y_1 = mask_y_0 + y_1 - y_0
        mask_x_0 = max(x_0 - box[0], 0)
        mask_x_1 = mask_x_0 + x_1 - x_0
        im_mask[y_0:y_1, x_0:x_1] = mask[
            mask_y_0 : mask_y_1, mask_x_0 : mask_x_1
        ]
        im_mask = im_mask[:, :, None]

        # OpenCV version 4.x
        contours, hierarchy = cv2.findContours(
            im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        image = cv2.drawContours(image, contours, -1, 25, 3)

        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
        ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='w', fontsize=12)
        ax.add_patch(rect)

    ax.imshow(image)
    plt.show()

display_objdetect_image(img, boxes, labels, scores, masks)
```



## Dataset (Train and validation)
The original pretrained Mask R-CNN model is from [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), compute mAP the same as [Detectron](https://github.com/facebookresearch/Detectron) on `coco_2014_minival` dataset from COCO, which is exactly equivalent to the `coco_2017_val` dataset.
<hr>

## Validation accuracy
Metric is COCO mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.
box mAP of 0.361, and mask mAP of 0.327.
<hr>

## Publication/Attribution
Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask R-CNN. IEEE International Conference on Computer Vision (ICCV), 2017.

Massa, Francisco and Girshick, Ross. maskrcnn-benchmark: Fast, modular reference implementation of Instance Segmentation and Object Detection algorithms in PyTorch. [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).
<hr>

## References
This model is converted from [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) with modifications in [repository](https://github.com/BowenBao/maskrcnn-benchmark/tree/onnx_stage).
<hr>

## License
Apache License v2.0
<hr>
