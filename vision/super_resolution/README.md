# SuperResolution

## Use cases
The Super Resolution machine learning model sharpens and upscales the input image to refine the details and improve quality.

## Description
Super Resolution uses efficient  [Sub-pixel convolutional layer](https://arxiv.org/abs/1609.05158) described for increasing spatial resolution within network tasks. By increasing pixel count, images are then clarified, sharpened, and upscaled without losing the input image’s content and characteristics. 

## Model

 |Model        |Download  |Checksum|Download (with sample test data)| ONNX version |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|MobileNet v2-1.0|    [240 KB](super_resolution_tutorial.onnx)  |[MD5](super_resolution-md5.txt)  |  [7.6 MB](super_resolution_test_image.gz) |  1.5.0  |

## Inference


### Input 
Image file can be jpg, png, and jpeg and its input sizes are dynamic. The inference was done using jpg image.

### Preprocessing
Images are resized into (224x224). The image is then split into ‘YCbCr’ color components: greyscale ‘Y’, blue-difference  ‘Cb’, and red-difference ‘Cr’. Once the greyscale Y component is extracted, it is then converted to tensor and used as the input image.

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
The model outputs a multidimensional array of pixels that are upscaled. Output shape is [batch_size,1,672,672]. 

### Postprocessing
Postprocessing involves converting the array of pixels into an image that is scaled to a higher resolution. The ‘YCbCr’ colors are then merged and reconstructed into the final output image. 

    final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")
    plt.imshow(final_img)


## Dataset
This example trains a super-resolution network on the [BSD300 Dataset](https://github.com/pytorch/examples/tree/master/super_resolution) , using crops from the 200 training images, and evaluating on crops of the 100 test images.

## Training
View the  [training notebook](https://github.com/pytorch/examples/tree/master/super_resolution) to understand details for parameters and network for SuperResolution

