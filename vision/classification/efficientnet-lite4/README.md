# EfficientNet-Lite4

## Use-Cases
EfficientNet-Lite4 is an image classification model that achieves state-of-the-art accuracy. It is designed to run on mobile CPU, GPU, and EdgeTPU devices, allowing for applications on mobile and loT, where computational resources are limited.

## Description
EfficientNet-Lite4 is a version of EfficentNet-Lite, where it is the largest variant, integer-only quantized model that produces the highest accuracy. ImageNet top-1 accuracy, while still running in real-time (e.g. 30ms/image) on a Pixel 4 CPU. 

## Model

 |Model        |Download | Download (with sample test data)|ONNX version|Opset version|
|-------------|:--------------|:--------------|:--------------|:--------------|
|EfficientNet-Lite4     | [51.9 MB](model/efficientnet-lite4.onnx)	  | [157 MB](efficientnet-lite4.tar.gz)|1.7.0|11|



<hr>

## Inference

### Input to model
Input to model has the image shape float32[1,224,224,3]. Inference was done using a jpg image.

### Preprocessing steps
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    import onnxruntime as rt
    import cv2
    import json
    
    #read the image
    fname = "image_file"
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #pre-process the image like mobilenet and resize it to 300x300
    img = pre_process_edgetpu(img, (224, 224, 3))
    plt.axis('off')
    plt.imshow(img)
    plt.show()
       
    def pre_process_edgetpu(img, dims):
        output_height, output_width, _ = dims
        img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
        img = center_crop(img, output_height, output_width)
        img = np.asarray(img, dtype='float32')
        img -= [127.0, 127.0, 127.0]
        img /= [128.0, 128.0, 128.0]
        return img
        
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


    def center_crop(img, out_height, out_width):
        height, width, _ = img.shape
        left = int((width - out_width) / 2)
        right = int((width + out_width) / 2)
        top = int((height - out_height) / 2)
        bottom = int((height + out_height) / 2)
        img = img[top:bottom, left:right]
        return img


### Output of model
Output of model is an inference score with array shape float32[1,1000] and label. 

### Postprocessing steps
    # load the model
    sess = rt.InferenceSession(MODEL + ".onnx")
    # run inference and print results
    results = sess.run(["Softmax:0"], {"images:0": img_batch})[0]
    result = reversed(results[0].argsort()[-5:])
    # result = np.argmax(results, axis=1)
    for r in result:
        print(r, labels[str(r-1)], results[0][r])
<hr>

## Dataset (Train and validation)
Model was trained on [COCO 2017 Train/Val annotations dataset](https://cocodataset.org/#download)
<hr>

## Validation
Refer to [efficientnet-lite4](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientnet-lite.ipynb) for details of how to use it and reproduce accuracy.
<hr>

## References
Original Pytorch to Onnx conversion [tutorial](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/efficientnet-lite.ipynb)
<hr>

## Contributors
 [Shirley Su](https://github.com/shirleysu8)
<hr>

## License
MIT License
<hr>
