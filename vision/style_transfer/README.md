# Fast Neural Style

## Use-cases
This artistic style transfer model mixes the content of an image with the style of another image. Below are examples of style transfer models. Each model transfers a unique style to a photograph of a door arch.  
|Model        |Mosaic  |Candy  | Rain Princess |Udnie  |Pointilism|
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|Style Images       |[<img src="images/style_images/mosaic.jpg">](images/style_images/mosaic.jpg)| [<img src="images/style_images/candy.jpg">](images/style_images/candy.jpg)|[<img src="images/style_images/rain_princess.jpg">](images/style_images/rain_princess.jpg)| [<img src="images/style_images/udnie.jpg">](images/style_images/udnie.jpg)| [<img src="images/style_images/pointilism.jpg">](images/style_images/pointilism.jpg)|
|Output Image      |[<img src="images/output_images/amber_mosaic.jpg">](images/style_images/amber_mosaic.jpg)| [<img src="images/output_images/amber_candy.jpg">](images/style_images/amber_candy.jpg)|[<img src="images/output_images/amber_rain_princess.jpg">](images/style_images/amber_rain_princess.jpg)| [<img src="images/output_images/amber_udnie.jpg">](images/style_images/amber_udnie.jpg)|[<img src="images/output_images/amber_pointilism.jpg">](images/style_images/amber_pointilism.jpg)|


## Description
The model uses the method described in [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) along with [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf).
  

## Model
 |Model        |Download  |Checksum| Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|Mosaic|ONNX Model download link with size  | MD5 checksum for the ONNX model| tar file containing ONNX model and synthetic test data (in .pb format)|ONNX version used for conversion| Opset version used for conversion|Accuracy values |
|Candy|ONNX Model download link with size  | MD5 checksum for the ONNX model| tar file containing ONNX model and synthetic test data (in .pb format)|ONNX version used for conversion| Opset version used for conversion|Accuracy values |
|Rain Princess|ONNX Model download link with size  | MD5 checksum for the ONNX model| tar file containing ONNX model and synthetic test data (in .pb format)|ONNX version used for conversion| Opset version used for conversion|Accuracy values | 
|Udnie|ONNX Model download link with size  | MD5 checksum for the ONNX model| tar file containing ONNX model and synthetic test data (in .pb format)|ONNX version used for conversion| Opset version used for conversion|Accuracy values |
|Pointilism|ONNX Model download link with size  | MD5 checksum for the ONNX model| tar file containing ONNX model and synthetic test data (in .pb format)|ONNX version used for conversion| Opset version used for conversion|Accuracy values |
<hr>

## Inference
Step by step instructions on how to use the pretrained model and link to an example notebook/code. This section should ideally contain:

### Input to model
Input to network (Example: 224x224 pixels in RGB)

### Preprocessing steps
Preprocessing required

### Output of model
Output of network

### Postprocessing steps
Post processing and meaning of output
<hr>

## Dataset (Train and validation)
This section should discuss datasets and any preparation steps if required.
<hr>

## Validation accuracy
Details of experiments leading to accuracy and comparison with the reference paper.
<hr>

## Training
Training details (preprocessing, hyperparameters, resources and environment) along with link to a training notebook (optional). 

Also clarify in case the model is not trained from scratch and include the source/process used to obtain the ONNX model.
<hr>

## Validation
Validation script/notebook used to obtain accuracy reported above along with details of how to use it and reproduce accuracy.
<hr>

## References
Link to references
<hr>

## Contributors
Contributors' name
<hr>

## License
Add license information
<hr>
