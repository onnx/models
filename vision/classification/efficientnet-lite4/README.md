<!--- SPDX-License-Identifier: MIT -->

# EfficientNet-Lite4

## Use Cases
EfficientNet-Lite4 is an image classification model that achieves state-of-the-art accuracy. It is designed to run on mobile CPU, GPU, and EdgeTPU devices, allowing for applications on mobile and loT, where computational resources are limited.

## Description
EfficientNet-Lite 4 is the largest variant and most accurate of the set of EfficientNet-Lite model. It is an integer-only quantized model that produces the highest accuracy of all of the EfficientNet models. It achieves 80.4% ImageNet top-1 accuracy, while still running in real-time (e.g. 30ms/image) on a Pixel 4 CPU.

## Model

 |Model        |Download | Download (with sample test data)|ONNX version|Opset version|
|-------------|:--------------|:--------------|:--------------|:--------------|
|EfficientNet-Lite4     | [51.9 MB](model/efficientnet-lite4-11.onnx)	  | [48.6 MB](model/efficientnet-lite4-11.tar.gz)|1.7.0|11|


### Source
Tensorflow EfficientNet-Lite4 => ONNX EfficientNet-Lite4

<hr>


## Inference

### Running Inference
The following steps show how to run the inference using onnxruntime.

    import onnxruntime as rt

    # load model
    sess = rt.InferenceSession(MODEL + ".onnx")
    # run inference
    results = sess.run(["Softmax:0"], {"images:0": img_batch})[0]


### Input to model
Input image to model is resized to shape `float32[1,224,224,3]`. The batch size is 1, with 224 x 224 height and width dimensions. The input is an RBG image that has 3 channels: red, green, and blue. Inference was done using a jpg image.

### Preprocessing steps
The following steps show how to preprocess the input image. For more details visit [this conversion notebook](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientnet-lite.ipynb).

    import numpy as np
    import math
    import matplotlib.pyplot as plt
    import onnxruntime as rt
    import cv2
    import json

    # load the labels text file
    labels = json.load(open("labels_map.txt", "r"))

    # set image file dimensions to 224x224 by resizing and cropping image from center
    def pre_process_edgetpu(img, dims):
        output_height, output_width, _ = dims
        img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
        img = center_crop(img, output_height, output_width)
        img = np.asarray(img, dtype='float32')
        # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
        img -= [127.0, 127.0, 127.0]
        img /= [128.0, 128.0, 128.0]
        return img

    # resize the image with a proportional scale
    def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
        height, width, _ = img.shape
        new_height = int(100. * out_height / scale)
        new_width = int(100. * out_width / scale)
        if height > width:
            w = new_width
            h = int(new_height * height / width)
        else:
            h = new_height
            w = int(new_width * width / height)
        img = cv2.resize(img, (w, h), interpolation=inter_pol)
        return img

    # crop the image around the center based on given height and width
    def center_crop(img, out_height, out_width):
        height, width, _ = img.shape
        left = int((width - out_width) / 2)
        right = int((width + out_width) / 2)
        top = int((height - out_height) / 2)
        bottom = int((height + out_height) / 2)
        img = img[top:bottom, left:right]
        return img

    # read the image
    fname = "image_file"
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pre-process the image like mobilenet and resize it to 224x224
    img = pre_process_edgetpu(img, (224, 224, 3))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    # create a batch of 1 (that batch size is buned into the saved_model)
    img_batch = np.expand_dims(img, axis=0)

### Output of model
Output of model is an inference score with array shape `float32[1,1000]`. The output references the `labels_map.txt` file which maps an index to a label to classify the type of image.

### Postprocessing steps
The following steps detail how to print the output results of the model.

    # load the model
    # Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
    # other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
    # based on the build flags) when instantiating InferenceSession.
    # Following code assumes NVIDIA GPU is available, you can specify other execution providers or don't include providers parameter
    # to use default CPU provider.
    sess = rt.InferenceSession(MODEL + ".onnx", providers=['CUDAExecutionProvider'])
    # run inference and print results
    results = sess.run(["Softmax:0"], {"images:0": img_batch})[0]
    result = reversed(results[0].argsort()[-5:])
    for r in result:
        print(r, labels[str(r)], results[0][r])
<hr>

## Dataset (Train and validation)
The model was trained using [COCO 2017 Train Images, Val Images, and Train/Val annotations](https://cocodataset.org/#download).
<hr>

## Validation
Refer to [efficientnet-lite4 conversion notebook](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientnet-lite.ipynb) for details of how to use it and reproduce accuracy.
<hr>

## References
Tensorflow to Onnx conversion [tutorial](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientnet-lite.ipynb). The Juypter Notebook references how to run an evaluation on the efficientnet-lite4 model and export it as a saved model. It also details how to convert the tensorflow model into onnx, and how to run its preprocessing and postprocessing code for the inputs and outputs.

Refer to this [paper](https://arxiv.org/abs/1905.11946) for more details on the model.
<hr>

## Contributors
 [Shirley Su](https://github.com/shirleysu8)
<hr>

## License
MIT License
<hr>
