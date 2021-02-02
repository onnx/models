<!--- SPDX-License-Identifier: Apache-2.0 -->

# Fully Convolutional Network (FCN)

## Description
FCNs are a model for real-time neural network for class-wise image segmentation. As the name implies, every weight layer in the network is convolutional. The final layer has the same height/width as the input image, making FCNs a useful tool for doing dense pixel-wise predictions without a significant amount of postprocessing. Being fully convolutional also provides great flexibility in the resolutions this model can handle.

This specific model detects 20 different [classes](dependencies/voc_classes.txt). The models have been pre-trained on the COCO train2017 dataset on this class subset.

## Model

| Model          | Download                              | Download (with sample test data)        | ONNX version | Opset version | Mean IoU |
|----------------|:--------------------------------------|:----------------------------------------|:-------------|:--------------|:--------|
| FCN ResNet-50  | [134 MB](model/fcn-resnet50-11.onnx)  | [213 MB](model/fcn-resnet50-11.tar.gz)  | 1.8.0        | 11 | 60.5% |
| FCN ResNet-101 | [207 MB](model/fcn-resnet101-11.onnx) | [281 MB](model/fcn-resnet101-11.tar.gz) | 1.8.0        | 11 | 63.7% |

### Source

 * PyTorch Torchvision FCN ResNet50 ==> ONNX FCN ResNet50
 * PyTorch Torchvision FCN ResNet101 ==> ONNX FCN ResNet101

## Inference

### Input
The input is expected to be an image with the shape `(N, 3, height, width)` where `N` is the number of images in the batch, and `height` and `width` are consistent across all images.

### Preprocessing
The images must be loaded in RGB with a range of `[0, 1]` per channel, then normalized per-image using `mean = [0.485, 0.456, 0.406]` and `std = [0.229, 0.224, 0.225]`.

This model can take images of different sizes as input. However, it is recommended that the images are resized such that the minimum size of either edge is 520.

The following code shows an example of how to preprocess a [demo image](dependencies/000000017968.jpg):

```python
from PIL import Image
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open('dependencies/000000017968.jpg')
img_data = preprocess(img).detach().cpu().numpy()
```

### Output of model
The model has two outputs, `("out", "aux")`. `"out"` is the main classifier and has shape `(N, 21, height, width)`. Each output pixel is one-hot encoded, i.e. `np.argmax(out[image, :, x, y])` is that pixel's predicted class. Class 0 is the background class.

`"aux"` is an auxilliary classifier with the same shape performing the same functionality. The difference between the two is that `"out"` sources features from last layer of the ResNet backbone, while `"aux"` sources features from the second-to-last layer.

### Postprocessing steps

The following code shows how to overlay the segmentation on the original image:

```python
from PIL import Image
from matplotlib.colors import hsv_to_rgb
import numpy as np
import cv2


classes = [line.rstrip('\n') for line in open('voc_classes.txt')]
num_classes = len(classes)

def get_palette():
    # prepare and return palette
    palette = [0] * num_classes * 3

    for hue in range(num_classes):
        if hue == 0: # Background color
            colors = (0, 0, 0)
        else:
            colors = hsv_to_rgb((hue / num_classes, 0.75, 0.75))

        for i in range(3):
            palette[hue * 3 + i] = int(colors[i] * 255)

    return palette

def colorize(labels):
    # generate colorized image from output labels and color palette
    result_img = Image.fromarray(labels).convert('P', colors=num_classes)
    result_img.putpalette(get_palette())
    return np.array(result_img.convert('RGB'))

def visualize_output(image, output):
    assert(image.shape[0] == output.shape[1] and \
           image.shape[1] == output.shape[2]) # Same height and width
    assert(output.shape[0] == num_classes)

    # get classification labels
    raw_labels = np.argmax(output, axis=0).astype(np.uint8)

    # comput confidence score
    confidence = float(np.max(output, axis=0).mean())

    # generate segmented image
    result_img = colorize(raw_labels)

    # generate blended image
    blended_img = cv2.addWeighted(image[:, :, ::-1], 0.5, result_img, 0.5, 0)

    result_img = Image.fromarray(result_img)
    blended_img = Image.fromarray(blended_img)

    return confidence, result_img, blended_img, raw_labels

conf, result_img, blended_img, raw_labels = visualize_output(orig_tensor, one_output)
```

## Model Creation

### Dataset (Train and validation)
The FCN models have been pretrained on the [COCO train2017 dataset](https://cocodataset.org/#download), using the subset of classes from Pascal VOC classes. See the [Torchvision Model Zoo](https://pytorch.org/docs/stable/torchvision/models.html) for more details.

### Training
Pretrained weights from the Torchvision Model Zoo were used instead of training these models from scratch. A [conversion notebook](dependencies/conversion.ipynb) is provided.

### Validation accuracy
Mean IoU (intersection over union) and global pixelwise accuracy are computed on the COCO val2017 dataset.
[Torchvision](https://pytorch.org/docs/stable/torchvision/models.html) reports these values as follows:
| Model          | mean IoU (%) | global pixelwise accuracy (%) |
|----------------|:-------------|:------------------------------|
| FCN ResNet 50  | 60.5         | 91.4                          |
| FCN ResNet 101 | 63.7         | 91.9                          |

If you have the [COCO val2017 dataset](https://cocodataset.org/#download) downloaded, you can confirm updated numbers using [the provided notebook](dependencies/validation_accuracy.ipynb):
| Model          | mean IoU | global pixelwise accuracy |
|----------------|:---------|:--------------------------|
| FCN ResNet 50  | 65.0     | 99.6                      |
| FCN ResNet 101 | 66.7     | 99.6                      |

The more conservative of the two estimates is used in the model files table.
<hr>

## References
Jonathan Long, Evan Shelhamer, Trevor Darrell; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 3431-3440

This model is converted from the [Torchvision Model Zoo](https://pytorch.org/docs/stable/torchvision/models.html), originally implemented by Francisco Moss [here](https://github.com/pytorch/vision/tree/master/torchvision/models/segmentation/fcn.py).

## Contributors
[Jack Duvall](https://github.com/duvallj)

## License
MIT License
