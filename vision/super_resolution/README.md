# SuperResolution

## Use cases
The Super Resolution machine learning model sharpens and upscales the input image to refine the details and improve quality.

## Description
Super Resolution uses efficient [sub-pixel convolution layer] (https://arxiv.org/abs/1609.05158) described for increasing spatial resolution within network tasks. By increasing pixel count, images are then clarified, sharpened, and upscaled without losing the input image’s content and characteristics. 

## Model
The below model allows for dynamic image size inputs
* Version 2:

 |Model        |Download  |Checksum|Download (with sample test data)| ONNX version |Opset version|Top-1 accuracy (%)|Top-5 accuracy (%)| 
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|MobileNet v2-1.0|    [13.6 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx)  |[MD5](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0-md5.txt)  |  [14.1 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz) |  1.2.1  | 7| 70.94    |     89.99           | 
<!--
|MobileNet v2-0.5|    [13.6 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-0.5/mobilenetv2-0.5.onnx)    |  [13.7 MB](https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-0.5/mobilenetv2-0.5.model)     |          |             |
-->

## Inference


### Input 
Image file can be jpg, png, and jpeg and its input sizes are dynamic. The inference was done using jpg image.

### Preprocessing
Images are resized into (224x224). The image is then split into ‘YCbCr’ color components: greyscale ‘Y’, blue-difference  ‘Cb’, and red-difference ‘Cr’. Once the greyscale Y component is extracted, it is then converted to tensor and used as the input image.

### Output
The model outputs a multidimensional array of pixels that are upscaled. Output shape is float32[batch_size,1,672,672]. 

### Postprocessing
Postprocessing involves converting the array of pixels into an image that is scaled to a higher resolution. The ‘YCbCr’ colors are then merged and reconstructed into the final output image. 

To do quick inference with the model, check out [Model Server](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md/#mobilenetv2-1.0_onnx).

## Dataset
This example trains a super-resolution network on the BSD300 dataset, using crops from the 200 training images, and evaluating on crops of the 100 test images.


## Validation accuracy
The accuracies obtained by the model on the validation set are mentioned above. The accuracies have been calculated on center cropped images with a maximum deviation of 1% (top-1 accuracy) from the paper.

## Training
View the training notebook to understand details for parameters and network for SuperResolution

## Validation
We used MXNet as framework with gluon APIs to perform validation. Use the notebook [imagenet_validation](../imagenet_validation.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.


## References
* **MobileNet-v2** Model from the paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

* [MXNet](http://mxnet.incubator.apache.org), [Gluon model zoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html), [GluonCV](https://gluon-cv.mxnet.io)

## Contributors
* [ankkhedia](https://github.com/ankkhedia) (Amazon AI)
* [abhinavs95](https://github.com/abhinavs95) (Amazon AI)

## License
Apache 2.0