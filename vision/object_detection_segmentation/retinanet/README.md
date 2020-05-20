# RetinanNet

## Description
[RetinaNet](https://github.com/NVIDIA/retinanet-examples) is a single-stage object detection model.

## Model

|Model        |Download  | Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|RetinaNet (ResNet101 backbone)|    [228.4 MB](model/retinanet.onnx)   | [143.7 MB](model/retinanet.tar.gz)    |  1.6.0  |9| mAP 0.376      |

## Inference
A sample script for ONNX model conversion and ONNXRuntime inference can be found [here](dependencies/retinanet-export.py).

### Input
The model expects mini-batches of 3-channel input images of shape (N x 3 x H x W), where N is batch size.

### Preprocessing
The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. The transformation should preferably happen at preprocessing.

The following code shows how to preprocess a NCHW tensor:

```python
import numpy as np

def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):  
         # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data
```


### Output

Model has 2 outputs:

Classification heads: List of classification heads.

Bounding box regression heads: List of bounding box heads.

### Postprocessing
The following script from [NVIDIA/retinanet-examples](https://github.com/NVIDIA/retinanet-examples/blob/0aba7724e42f5b654d8171a6cac8b54e07fb8206/retinanet/model.py#L141) shows how to:
 1) Generate anchor boxes.
 2) Decode and then filter box predictions from at most 1k top-scoring predictions per level, with confidence threshold of 0.05.
 3) Apply non-maximum suppression on anchor boxes to get ground-truth boxes, scores, and labels.

```python
import torch
from retinanet.box import generate_anchors, decode, nms

def detection_postprocess(image, cls_heads, box_heads):
	# Inference post-processing
	anchors = {}
	decoded = []
	
	for cls_head, box_head in zip(cls_heads, box_heads):
	    # Generate level's anchors
	    stride = image.shape[-1] // cls_head.shape[-1]
	    if stride not in anchors:
	        anchors[stride] = generate_anchors(stride, ratio_vals=[1.0, 2.0, 0.5],
	                                           scales_vals=[4 * 2 ** (i / 3) for i in range(3)])
	    # Decode and filter boxes
	    decoded.append(decode(cls_head, box_head, stride,
	                          threshold=0.05, top_n=1000, anchors=anchors[stride]))
	
	# Perform non-maximum suppression
	decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
	# NMS threshold = 0.5
	scores, boxes, labels = nms(*decoded, nms=0.5, ndetections=100)
	return scores, boxes, labels
	

scores, boxes, labels = detection_postprocess(input_image, cls_heads, box_heads)

```

## Dataset
Model backbone is initialized using pre-trained [pytorch/vision ResNet101](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet101) model. This model is pre-trained on the ```ImageNet``` dataset.

## Validation accuracy
The accuracies obtained by the model on the validation set is provided by [NVIDIA/retinanet-examples](https://github.com/NVIDIA/retinanet-examples).
Metric is COCO mAP (averaged over IoU of 0.5:0.95), computed over [COCO 2017](http://cocodataset.org/#detection-2017) val data.

## Publication/Attribution
[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár. arXiv, 2017.

## References
This model is converted directly from [NVIDIA/retinanet-examples](https://github.com/NVIDIA/retinanet-examples).
<hr>

## License
BSD 3-Clause "New" or "Revised" License