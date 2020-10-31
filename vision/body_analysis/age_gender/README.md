# Age and Gender Classification using Convolutional Neural Networks

## Description
Rothe et al. in their [paper](https://data.vision.ee.ethz.ch/cvl/publications/papers/proceedings/eth_biwi_01229.pdf) propose a deep learning solution to age estimation from a single face image without the use of facial landmarks and introduce the [IMDB-WIKI dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/), the largest public dataset of face images with age and gender labels. If the real age estimation research spans over decades, the study of apparent age estimation or the age as perceived by other humans from a face image is a recent endeavor. Rothe et al. tackle both tasks with their convolutional neural networks (CNNs) of VGG-16 architecture which are pre-trained on ImageNet for image classification. They pose the age estimation problem as a deep classification problem followed by a softmax expected value refinement. The key factors of our solution are: deep learned models from large data, robust face alignment, and expected value formulation for age regression. They validate their methods on standard benchmarks and achieve state-of-the-art results for both real and apparent age estimation.

## Models
|Model|Download|ONNX version|Opset version|
|-------------|:--------------|:--------------|:--------------|
|VGG_ILSVRC_16_layers_Age|[513 MB](models/VGG_ILSVRC_16_layers_Age.onnx)| 1.5 | 5 |
|VGG_ILSVRC_16_layers_Gender|[512 MB](models/VGG_ILSVRC_16_layers_Gender.onnx)| 1.5 | 5 |

## References
* Levi et al. - [Age and Gender Classification Using Convolutional Neural Networks](https://talhassner.github.io/home/publication/2015_CVPR) (**NOT compatible with ONNX**).
* Rothe et al. - [IMDB-WIKI – 500k+ face images with age and gender labels](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).

## Contributors
[asiryan](https://github.com/asiryan)

## License
Apache 2.0
