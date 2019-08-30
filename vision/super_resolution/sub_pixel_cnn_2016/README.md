# Super Resolution

## Use cases
The Super Resolution machine learning model sharpens and upscales the input image to refine the details and improve quality.

## Description
Super Resolution uses efficient  [Sub-pixel convolutional layer](https://arxiv.org/abs/1609.05158) described for increasing spatial resolution within network tasks. By increasing pixel count, images are then clarified, sharpened, and upscaled without losing the input image’s content and characteristics. 

## Model

 |Model      |Download  |Checksum|Download (with sample test data)| ONNX version | Opset Version 
|-------------|:--------------|:--------------|:--------------|:--------------| :------------|
|Super_Resolution|    [240 KB](model/super_resolution.onnx)  |[MD5](super_resolution-md5.txt)  |  [7.6 MB](model/super_resolution_test_image.tar) |  1.5.0  | 10

## Inference


### Input 
Image input sizes are dynamic. The inference was done using jpg image. 

### Preprocessing
Images are resized into (224x224). The image format is changed into YCbCr with color components: greyscale ‘Y’, blue-difference  ‘Cb’, and red-difference ‘Cr’. Once the greyscale Y component is extracted, it is then passed through the super resolution model and upscaled. 
  
    from PIL import Image
    from resizeimage import resizeimage
    import numpy as np
    orig_img = Image.open('IMAGE_FILE_PATH')
    img = resizeimage.resize_cover(orig_img, [224,224], validate=False)
    img_ycbcr = img.convert('YCbCr')
    img_y_0, img_cb, img_cr = img_ycbcr.split()
    img_ndarray = np.asarray(img_y_0)
    img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
    img_5 = img_4.astype(np.float32) / 255.0
    img_5
  

### Output
The model outputs a multidimensional array of pixels that are upscaled. Output shape is [batch_size,1,672,672]. The second dimension is one because only the (Y) intensity channel was passed into the super resolution model and upscaled. 

### Postprocessing
Postprocessing involves converting the array of pixels into an image that is scaled to a higher resolution. The color channels (Cb, Cr) are also scaled to a higher resolution using bicubic interpolation. Then the color channels are combined and converted back to RGB format, producing the final output image. 

    final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")
    plt.imshow(final_img)


## Dataset
This model is trained on the [BSD300 Dataset](https://github.com/pytorch/examples/tree/master/super_resolution), using crops from the 200 training images. 

## Training
View the  [training notebook](https://github.com/pytorch/examples/tree/master/super_resolution) to understand details for parameters and network for SuperResolution.

