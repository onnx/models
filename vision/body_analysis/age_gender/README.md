<!--- SPDX-License-Identifier: Apache-2.0 -->

# Age and Gender Classification using Convolutional Neural Networks

## Description
Automatic age and gender classification has become relevant to an increasing amount of applications, particularly since the rise of social platforms and social media. Nevertheless, performance of existing methods on real-world images is still significantly lacking, especially when compared to the tremendous leaps in performance recently reported for the related task of face recognition.

## Models
| Model (Caffe) | Download | ONNX version | Opset version | Dataset |
|:-------------|:--------------|:--------------|:--------------|:--------------|
| [googlenet_age_adience](https://drive.google.com/drive/folders/1GeLTHzHALgTYFj2Q9o5aWdztA9WzoErx?usp=sharing) | [23 MB](models/age_googlenet.onnx) | 1.6 | 11 | Adience |
| [googlenet_gender_adience](https://drive.google.com/drive/folders/1r0GroTfsF7VpLhcS3IxU-LmAh6rI6vbQ?usp=sharing) | [23 MB](models/gender_googlenet.onnx)| 1.6 | 11 | Adience |
| [vgg_ilsvrc_16_age_chalearn_iccv2015](https://drive.google.com/drive/folders/1wE4_sj-UBumkjDK9mtfaO9eUan_z44cY?usp=sharing) | [513 MB](models/vgg_ilsvrc_16_age_chalearn_iccv2015.onnx) | 1.6 | 11 | ChaLearn LAP 2015 |
| [vgg_ilsvrc_16_age_imdb_wiki](https://drive.google.com/drive/folders/14wckle-MbnN10xzdzgF464bMnlM-dd5-?usp=sharing) | [513 MB](models/vgg_ilsvrc_16_age_imdb_wiki.onnx)| 1.6 | 11 | IMDB-WIKI |
| [vgg_ilsvrc_16_gender_imdb_wiki](https://drive.google.com/drive/folders/16Z1r7GEXCsJG_384VsjlNxOFXbxcXrqM?usp=sharing) | [512 MB](models/vgg_ilsvrc_16_gender_imdb_wiki.onnx)| 1.6 | 11 | IMDB-WIKI |

## Inference
### GoogleNet
Input tensor is `1 x 3 x height x width` with mean values `104, 117, 123`. Input image have to be previously resized to `224 x 224` pixels and converted to `BGR` format.
Run [levi_googlenet.py](levi_googlenet.py) python script example.

### VGG-16
Input tensor is `1 x 3 x height x width`, which values are in range of `[0, 255]`. Input image have to be previously resized to `224 x 224` pixels and converted to `BGR` format.
Run [rothe_vgg.py](rothe_vgg.py) python script example.

## References
* Levi et al. - [Age and Gender Classification Using Convolutional Neural Networks](https://talhassner.github.io/home/publication/2015_CVPR).
* Rothe et al. - [IMDB-WIKI – 500k+ face images with age and gender labels](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).
* Lapuschkin et al. - [Understanding and Comparing Deep Neural Networks for Age and Gender Classification](https://github.com/sebastian-lapuschkin/understanding-age-gender-deep-learning-models).
* Caffe to ONNX: [unofficial converter](https://github.com/asiryan/caffe-onnx).

## Contributors
Valery Asiryan ([asiryan](https://github.com/asiryan))

## License
Apache 2.0
