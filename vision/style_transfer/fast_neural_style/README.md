<!--- SPDX-License-Identifier: BSD-3-Clause -->

# Fast Neural Style Transfer

## Use-cases
This artistic style transfer model mixes the content of an image with the style of another image. Examples of the styles can be seen [in this PyTorch example](https://github.com/pytorch/examples/tree/master/fast_neural_style#models).


## Description
The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf).


## Model
 |Model        |Download  |Download (with sample test data)|ONNX version|Opset version|
|-------------|:--------------|:--------------|:--------------|:--------------|
|Mosaic|[6.6 MB](model/mosaic-9.onnx)  | [7.2 MB](model/mosaic-9.tar.gz)|1.4|9|
|Candy|[6.6 MB](model/candy-9.onnx)  | [7.2 MB](model/candy-9.tar.gz)|1.4|9|
|Rain Princess|[6.6 MB](model/rain-princess-9.onnx)  |[7.2 MB](model/rain-princess-9.tar.gz)|1.4|9|
|Udnie|[6.6 MB](model/udnie-9.onnx)  | [7.2 MB](model/udnie-9.tar.gz)|1.4|9|
|Pointilism|[6.6 MB](model/pointilism-9.onnx)  | [7.2 MB](model/pointilism-9.tar.gz)|1.4|9|
|Mosaic|[6.6 MB](model/mosaic-8.onnx)  | [7.2 MB](model/mosaic-8.tar.gz)|1.4|8|
|Candy|[6.6 MB](model/candy-8.onnx)  | [7.2 MB](model/candy-8.tar.gz)|1.4|8|
|Rain Princess|[6.6 MB](model/rain-princess-8.onnx)  |[7.2 MB](model/rain-princess-8.tar.gz)|1.4|8|
|Udnie|[6.6 MB](model/udnie-8.onnx)  | [7.2 MB](model/udnie-8.tar.gz)|1.4|8|
|Pointilism|[6.6 MB](model/pointilism-8.onnx)  | [7.2 MB](model/pointilism-8.tar.gz)|1.4|8|
<hr>

## Inference
Refer to [style-transfer-ort.ipynb](dependencies/style-transfer-ort.ipynb) for detailed preprocessing and postprocessing.

### Input to model
The input to the model are 3-channel RGB images. The images have to be loaded in a range of [0, 255]. If running into memory issues, try resizing the image by increasing the scale number.

### Preprocessing steps
```
from PIL import Image
import numpy as np

# loading input and resize if needed
image = Image.open("PATH TO IMAGE")
size_reduction_factor = 1
image = image.resize((int(image.size[0] / size_reduction_factor), int(image.size[1] / size_reduction_factor)), Image.ANTIALIAS)

# Preprocess image
x = np.array(image).astype('float32')
x = np.transpose(x, [2, 0, 1])
x = np.expand_dims(x, axis=0)
```

### Output of model
The converted ONNX model outputs a NumPy float32 array of shape [1, 3, ‘height’, ‘width’]. The height and width of the output image are the same as the height and width of the input image.

### Postprocessing steps
```
result = np.clip(result, 0, 255)
result = result.transpose(1,2,0).astype("uint8")
img = Image.fromarray(result)
```
<hr>

## Dataset (Train and validation)
The original fast neural style model is from [pytorch/examples/fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style). All models are trained using the [COCO 2014 Training images dataset](http://cocodataset.org/#download) [80K/13GB].
<hr>

## Training
Refer to [pytorch/examples/fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style) for training details in PyTorch. Refer to [conversion.ipynb](dependencies/conversion.ipynb) to learn how the PyTorch models are converted to ONNX format.
<hr>


## References
Original style transfer model in PyTorch: <https://github.com/pytorch/examples/tree/master/fast_neural_style>
<hr>

## Contributors
[Jennifer Wang](https://github.com/jennifererwangg)
<hr>

## License
BSD-3-Clause
<hr>
