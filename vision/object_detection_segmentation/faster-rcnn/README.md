# Faster R-CNN

## Description
This model is a real-time neural network for object detection that detects 80 different [classes](coco_classes.txt).

## Model

|Model        |Download  | Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|Faster R-CNN R-50-FPN      |[167.3 MB](model/FasterRCNN-10.onnx) |[158.0 MB](model/FasterRCNN-10.tar.gz) |1.5 |10 |mAP of 0.35 |



<hr>

## Inference

### Input to model
Image shape `(3x'height'x'width')`

### Preprocessing steps
The images have to be loaded in to a range of [0, 255], resized, converted to BGR and then normalized using mean = [102.9801, 115.9465, 122.7717]. The transformation should preferably happen at preprocessing.

This model can take images of different sizes as input. However, to achieve best performance, it is recommended to resize the image such that both height and width are within the range of [800, 1333], and then pad the image with zeros such that both height and width are divisible by 32.

The following code shows how to preprocess the [demo image](demo.jpg):

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

img = Image.open('demo.jpg')
img_data = preprocess(img)
```

### Output of model
The model has 3 outputs.

boxes: `('nbox'x4)`, in `(xmin, ymin, xmax, ymax)`.

labels: `('nbox')`.

scores: `('nbox')`.

### Postprocessing steps

The following code shows how to patch the original image with detections and class annotations, filtered by scores:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

classes = [line.rstrip('\n') for line in open('coco_classes.txt')]

def display_objdetect_image(image, boxes, labels, scores, score_threshold=0.7):
    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes /= ratio

    _, ax = plt.subplots(1, figsize=(12,9))
    image = np.array(image)
    ax.imshow(image)

    # Showing boxes with score > 0.7
    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
            ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='w', fontsize=12)
            ax.add_patch(rect)
    plt.show()

display_objdetect_image(img, boxes, labels, scores)
```



## Dataset (Train and validation)
The original pretrained Faster R-CNN model is from [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), compute mAP the same as [Detectron](https://github.com/facebookresearch/Detectron) on `coco_2014_minival` dataset from COCO, which is exactly equivalent to the `coco_2017_val` dataset.
<hr>

## Validation accuracy
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.
mAP of 0.353
<hr>

## Publication/Attribution
Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. Conference on Neural Information Processing Systems (NIPS), 2015.

Massa, Francisco and Girshick, Ross. maskrcnn-benchmark: Fast, modular reference implementation of Instance Segmentation and Object Detection algorithms in PyTorch. [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).
<hr>

## References
This model is converted from [facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) with modifications in [repository](https://github.com/BowenBao/maskrcnn-benchmark/tree/onnx_stage).
<hr>

## License
MIT License
<hr>
