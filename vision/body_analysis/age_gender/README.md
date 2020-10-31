# Age and Gender Classification using Convolutional Neural Networks

## Description
Rothe et al. in their [paper](https://data.vision.ee.ethz.ch/cvl/publications/papers/proceedings/eth_biwi_01229.pdf) propose a deep learning solution to age estimation from a single face image without the use of facial landmarks and introduce the [IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/), the largest public dataset of face images with age and gender labels. If the real age estimation research spans over decades, the study of apparent age estimation or the age as perceived by other humans from a face image is a recent endeavor. Rothe et al. tackle both tasks with their convolutional neural networks (CNNs) of VGG-16 architecture which are pre-trained on ImageNet for image classification. They pose the age estimation problem as a deep classification problem followed by a softmax expected value refinement. The key factors of our solution are: deep learned models from large data, robust face alignment, and expected value formulation for age regression. They validate their methods on standard benchmarks and achieve state-of-the-art results for both real and apparent age estimation.

## Model
| Model (Caffe) | Download | ONNX version | Opset version | Dataset |
|:-------------|:--------------|:--------------|:--------------|:--------------|
| [vgg_ilsvrc_16_age_chalearn_iccv2015](https://drive.google.com/drive/folders/1wE4_sj-UBumkjDK9mtfaO9eUan_z44cY?usp=sharing) | [513 MB](https://drive.google.com/file/d/1V75U1kUJ0udBLs6bg3lGqBk3ym8q9guV/view?usp=sharing) | 1.5 | 5 | ChaLearn LAP 2015 |
| [vgg_ilsvrc_16_age_imdb_wiki](https://drive.google.com/drive/folders/14wckle-MbnN10xzdzgF464bMnlM-dd5-?usp=sharing) | [513 MB](https://drive.google.com/file/d/1ECle8EvsXiIid_vMa1_vwMJk6abhzrPF/view?usp=sharing)| 1.5 | 5 | IMDB-WIKI |
| [vgg_ilsvrc_16_gender_imdb_wiki](https://drive.google.com/drive/folders/16Z1r7GEXCsJG_384VsjlNxOFXbxcXrqM?usp=sharing) | [512 MB](https://drive.google.com/file/d/1epLM5ghucLcnGZg-NCIf1r16lotN004I/view?usp=sharing)| 1.5 | 5 | IMDB-WIKI |

## Inference
Input tensor ```1 x 3 x height x width```, which values are in range of ```[0, 255]```. Input image have to be previously resized to ```224 x 224``` pixels and converted to **BGR** format.

## References
* Levi et al. - [Age and Gender Classification Using Convolutional Neural Networks](https://talhassner.github.io/home/publication/2015_CVPR) (**NOT compatible with ONNX**).
* Rothe et al. - [IMDB-WIKI – 500k+ face images with age and gender labels](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).

## Contributors
Asiryan Valery ([asiryan](https://github.com/asiryan))

## License
Apache 2.0
