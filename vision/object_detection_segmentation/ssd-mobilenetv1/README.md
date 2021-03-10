<!--- SPDX-License-Identifier: MIT -->

# SSD-MobilenetV1

## Description

SSD-MobilenetV1 is an object detection model that uses a Single Shot MultiBox Detector (SSD) approach to predict object classes for boundary boxes.

SSD is a CNN that enables the model to only need to take one single shot to detect multiple objects in an image, and MobileNet is a CNN base network that provides high-level features for object detection. The combination of these two model frameworks produces an efficient, high-accuracy detection model that requires less computational cost.

The SSD-MobilenetV1 is suitable for mobile and embedded vision applications.

## Model

|Model        |Download  | Download (with sample test data)|ONNX version|Opset version|
|-------------|:--------------|:--------------|:--------------|:--------------|
|SSD-MobilenetV1       | [29.3 MB](model/ssd_mobilenet_v1_10.onnx)  |[27.9 MB](model/ssd_mobilenet_v1_10.tar.gz) |1.7.0 | 10 |

### Source
Tensorflow SSD-MobileNetV1 ==> ONNX SSD-MobileNetV1

## Inference

### Running inference
Refer to this [conversion and inference notebook](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/ConvertingSSDMobilenetToONNX.ipynb) for more details on how to inference this model using onnxruntime and define environment variables for the model.

    import onnxruntime as rt

    # load model and run inference
    sess = rt.InferenceSession(os.path.join(WORK, MODEL + ".onnx"))
    result = sess.run(outputs, {"image_tensor:0": img_data})
    num_detections, detection_boxes, detection_scores, detection_classes = result

    # print number of detections
    print(num_detections)
    print(detection_classes)

    # produce outputs in this order
    outputs = ["num_detections:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0"]


### Input
This model does not require fixed image dimensions. Input batch size is 1, with 3 color channels. Image has these variables: `(batch_size, height, width, channels)`.

### Preprocessing
The following code shows how preprocessing is done. For more information and an example on how preprocessing is done, please visit the [tf2onnx conversion and inference notebook](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/ConvertingSSDMobilenetToONNX.ipynb) for this model.

    import numpy as np
    from PIL import Image, ImageDraw, ImageColor
    import math
    import matplotlib.pyplot as plt

    # open and display image file
    img = Image.open("image file")
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    # reshape the flat array returned by img.getdata() to HWC and than add an additial
    dimension to make NHWC, aka a batch of images with 1 image in it
    img_data = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
    img_data = np.expand_dims(img_data.astype(np.uint8), axis=0)

### Output

It outputs the image with boundary boxes and labels. The full list of classes can be found in the [COCO dataset](https://cocodataset.org/#home).

Given each batch of images, the model returns 4 tensor arrays:

`num_detections`: the number of detections.

`detection_boxes`: a list of bounding boxes. Each list item describes a box with top, left, bottom, right relative to the image size.

`detection_scores`: the score for each detection with values between 0 and 1 representing probability that a class was detected.

`detection_classes`: Array of 10 integers (floating point values) indicating the index of a class label from the COCO class.



### Postprocessing

    # draw boundary boxes and label for each detection
    def draw_detection(draw, d, c):
        width, height = draw.im.size
        # the box is relative to the image size so we multiply with height and width to get pixels
        top = d[0] * height
        left = d[1] * width
        bottom = d[2] * height
        right = d[3] * width
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
        right = min(width, np.floor(right + 0.5).astype('int32'))
        label = coco_classes[c]
        label_size = draw.textsize(label)
        if top - label_size[1] >= 0:
            text_origin = tuple(np.array([left, top - label_size[1]]))
        else:
            text_origin = tuple(np.array([left, top + 1]))
        color = ImageColor.getrgb("red")
        thickness = 0
        draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness],
        outline=color)
        draw.text(text_origin, label, fill=color), font=font)

    # loop over the results - each returned tensor is a batch
    batch_size = num_detections.shape[0]
    draw = ImageDraw.Draw(img)
    for batch in range(0, batch_size):
        for detection in range(0, int(num_detections[batch])):
            c = detection_classes[batch][detection]
            d = detection_boxes[batch][detection]
            draw_detection(draw, d, c)

    # show image file with object detection boundary boxes and labels
    plt.figure(figsize=(80, 40))
    plt.axis('off')
    plt.imshow(img)
    plt.show()

## Model Creation

### Dataset (Train and validation)

The model was trained using [MS COCO 2017 Train Images, Val Images, and Train/Val annotations](https://cocodataset.org/#download).

### Training

Training details for the SSD-MobileNet model's preprocessing is found in this [tutorial notebook](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/ConvertingSSDMobilenetToONNX.ipynb).
The notebook also details how the ONNX model was converted.

### References
Tensorflow to ONNX conversion [tutorial](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/ConvertingSSDMobilenetToONNX.ipynb). The notebook references how to run an evaluation on the SSD-MobilenetV1 model and export it as a saved model. It also details how to convert the tensorflow model into onnx, and how to run its preprocessing and postprocessing code for the inputs and outputs.


## Contributors
[Shirley Su](https://github.com/shirleysu8)

## License
MIT License
<hr>
