<!--- SPDX-License-Identifier: Apache-2.0 -->

# Single Stage Detector

## Description
This model is a real-time neural network for object detection that detects 80 different classes.

## Model

|Model        |Download  | Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|SSD       |[80.4 MB](model/ssd-10.onnx) | [78.5 MB](model/ssd-10.tar.gz) |1.5 |10 |mAP of 0.195 |
|SSD       |[77.6 MB](model/ssd-12.onnx) | [86.4 MB](model/ssd-12.tar.gz) |1.9 |12 |mAP of 0.1898 |
|SSD-int8|[20 MB](model/ssd-12-int8.onnx) | [31 MB](model/ssd-12-int8.tar.gz) |1.9 |12 |mAP of 0.1892 |
|SSD-qdq|[20 MB](model/ssd-12-qdq.onnx) | [26 MB](model/ssd-12-qdq.tar.gz) |1.9 |12 |mAP of 0.1863 |

> Compared with the fp32 SSD, SSD-int8's mAP drop ratio is 0.32% and performance improvement is 3.49x.
>
> **Note**
>
> The performance depends on the test hardware. Performance data here is collected with Intel® Xeon® Platinum 8280 Processor, 1s 4c per instance, CentOS Linux 8.3, data batch size is 1.

<hr>

## Inference

### Input to model
Image shape `(1x3x1200x1200)`

### Preprocessing steps
The images have to be loaded in to a range of [0, 1], resized to (1200, 1200) with bilinear interpolation and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. The transformation should preferrably happen at preprocessing.

The following code shows how to preprocess a NCHW tensor:

```python
import numpy as np
from PIL import Image

def preprocess(img_path):
    input_shape = (1, 3, 1200, 1200)
    img = Image.open(img_path)
    img = img.resize((1200, 1200), Image.BILINEAR)
    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:,i,:,:] = (img_data[:,i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data
```

### Output of model
The model has 3 outputs.
boxes: `(1x'nbox'x4)`
labels: `(1x'nbox')`
scores: `(1x'nbox')`

<!-- ### Postprocessing steps
Post processing and meaning of output
<hr> -->

## Dataset (Train and validation)
The SSD model was trained on 2017 COCO train data set - using mlperf/training/single_stage_detector repo , compute mAP on 2017 COCO val data set.
<hr>

## Validation accuracy
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.
mAP of 0.195
<hr>

## Publication/Attribution
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is ResNet34 pretrained on ILSVRC 2012 (from torchvision). Modifications to the backbone networks: remove conv_5x residual blocks, change the first 3x3 convolution of the conv_4x block from stride 2 to stride1 (this increases the resolution of the feature map to which detector heads are attached), attach all 6 detector heads to the output of the last conv_4x residual block. Thus detections are attached to 38x38, 19x19, 10x10, 5x5, 3x3, and 1x1 feature maps. Convolutions in the detector layers are followed by batch normalization layers.
<hr>

## Quantization
SSD-int8 and SSD-qdq are obtained by quantizing fp32 SSD model. We use [Intel® Neural Compressor](https://github.com/intel/neural-compressor) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/object_detection/onnx_model_zoo/ssd/quantization/ptq/README.md) to understand how to use Intel® Neural Compressor for quantization.

### Environment
onnx: 1.9.0 
onnxruntime: 1.8.0

### Prepare model
```shell
wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/ssd/model/ssd-12.onnx
```

### Model quantize
Make sure to specify the appropriate dataset path in the configuration file.
```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=ssd.yaml \
                   --output_model=path/to/save
```
<hr>

## References
* This model is converted from mlperf/inference [repository](https://github.com/mlperf/inference/tree/master/others/cloud/single_stage_detector) with modifications in [repository](https://github.com/BowenBao/inference/tree/master/cloud/single_stage_detector/pytorch).

* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)
<hr>

## Contributors
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [yuwenzho](https://github.com/yuwenzho) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)
<hr>

## License
Apache License 2.0
<hr>
