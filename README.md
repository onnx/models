# ONNX Model Zoo

Open Neural Network Exchange (ONNX) is an open standard format for representing machine learning models. ONNX is supported by a community of partners who have implemented it in many frameworks and tools.

The ONNX Model Zoo is a collection of pre-trained, state-of-the-art models in the [ONNX](http://onnx.ai) format contributed by community members like you. Accompanying each model are [Jupyter notebooks](http://jupyter.org) for model training and running inference with the trained model. The notebooks are written in Python and include links to the training dataset as well as references to the original paper that describes the model architecture. 

## Models
#### Read the [Usage](#usage-) section below for more details on the file formats in the ONNX Model Zoo (.onnx, .pb, .npz) and starter Python code for validating your ONNX model using test data.

* [Image Classification](#image_classification)
* [Object Detection & Image Segmentation](#object_detection)
* [Body, Face & Gesture Analysis](#body_analysis)
* [Image Manipulation](#image_manipulation)
* [Speech & Audio Processing](#speech)
* [Machine Comprehension](#machine_comprehension)
* [Machine Translation](#machine_translation)
* [Language Modelling](#language)
* [Visual Question Answering & Dialog](#visual_qna)
* [Other interesting models](#others)

### Image Classification <a name="image_classification"/>
This collection of models take images as input, then classifies the major objects in the images into 1000 object categories such as keyboard, mouse, pencil, and many animals.

|Model Class |Reference |Description |
|-|-|-|
|<b>[MobileNet](vision/classification/mobilenet)</b>|[Sandler et al.](https://arxiv.org/abs/1801.04381)|Light-weight deep neural network best suited for mobile and embedded vision applications. <br>Top-5 error from paper - ~10%|
|<b>[ResNet](vision/classification/resnet)</b>|[He et al.](https://arxiv.org/abs/1512.03385)|A CNN model (up to 152 layers). Uses shortcut connections to achieve higher accuracy when classifying images. <br> Top-5 error from paper - ~3.6%|
|<b>[SqueezeNet](vision/classification/squeezenet)</b>|[Iandola et al.](https://arxiv.org/abs/1602.07360)|A light-weight CNN model providing AlexNet level accuracy with 50x fewer parameters. <br>Top-5 error from paper - ~20%|
|<b>[VGG](vision/classification/vgg)</b>|[Simonyan et al.](https://arxiv.org/abs/1409.1556)|Deep CNN model(up to 19 layers). Similar to AlexNet but uses multiple smaller kernel-sized filters that provides more accuracy when classifying images. <br>Top-5 error from paper - ~8%|
|<b>[AlexNet](vision/classification/alexnet)</b>|[Krizhevsky et al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)|A Deep CNN model (up to 8 layers) where the input is an image and the output is a vector of 1000 numbers. <br> Top-5 error from paper - ~15%|
|<b>[GoogleNet](vision/classification/inception_and_googlenet/googlenet)</b>|[Szegedy et al.](https://arxiv.org/pdf/1409.4842.pdf)|Deep CNN model(up to 22 layers). Comparatively smaller and faster than VGG and more accurate in detailing than AlexNet. <br> Top-5 error from paper - ~6.7%|
|<b>[CaffeNet](vision/classification/caffenet)</b>|[Krizhevsky et al.]( https://ucb-icsi-vision-group.github.io/caffe-paper/caffe.pdf)|Deep CNN variation of AlexNet for Image Classification in Caffe where the max pooling precedes the local response normalization (LRN) so that the LRN takes less compute and memory.|
|<b>[RCNN_ILSVRC13](vision/classification/rcnn_ilsvrc13)</b>|[Girshick et al.](https://arxiv.org/abs/1311.2524)|Pure Caffe implementation of R-CNN for image classification. This model uses localization of regions to classify and extract features from images.|
|<b>[DenseNet-121](vision/classification/densenet-121)</b>|[Huang et al.](https://arxiv.org/abs/1608.06993)|Model that has every layer connected to every other layer and passes on its own feature providing strong gradient flow and more diversified features.|
|<b>[Inception_V1](vision/classification/inception_and_googlenet/inception_v1)</b>|[Szegedy et al.](https://arxiv.org/abs/1409.4842)|This model is same as GoogLeNet, implemented through Caffe2 that has improved utilization of the computing resources inside the network and helps with the vanishing gradient problem. <br> Top-5 error from paper - ~6.7%|
|<b>[Inception_V2](vision/classification/inception_and_googlenet/inception_v2)</b>|[Szegedy et al.](https://arxiv.org/abs/1512.00567)|Deep CNN model for Image Classification as an adaptation to Inception v1 with batch normalization. This model has reduced computational cost and improved image resolution compared to Inception v1. <br> Top-5 error from paper ~4.82%|
|<b>[ShuffleNet](vision/classification/shufflenet)</b>|[Zhang et al.](https://arxiv.org/abs/1707.01083)|Extremely computation efficient CNN model that is designed specifically for mobile devices. This model greatly reduces the computational cost and provides a ~13x speedup over AlexNet on ARM-based mobile devices. Compared to MobileNet, ShuffleNet achieves superior performance by a significant margin due to it's efficient structure. <br> Top-1 error from paper - ~7.8%|
|<b>[ZFNet-512](vision/classification/zfnet-512)</b>|[Zeiler et al.](https://arxiv.org/abs/1311.2901)|Deep CNN model (up to 8 layers) that increased the number of features that the network is capable of detecting that helps to pick image features at a finer level of resolution. <br> Top-5 error from paper - ~14.3%|
<hr>

#### Domain-based Image Classification <a name="domain_based_image"/>
This subset of models classify images for specific domains and datasets.

|Model Class |Reference |Description |
|-|-|-|
|<b>[MNIST-Handwritten Digit Recognition](vision/classification/mnist)</b>|[Convolutional Neural Network with MNIST](https://github.com/Microsoft/CNTK/blob/master/Tutorials/CNTK_103D_MNIST_ConvolutionalNeuralNetwork.ipynb)	|Deep CNN model for handwritten digit identification|
<hr>

### Object Detection & Image Segmentation <a name="object_detection"/>
Object detection models detect the presence of multiple objects in an image and segment out areas of the image where the objects are detected. Semantic segmentation models partition an input image by labeling each pixel into a set of pre-defined categories.

|Model Class |Reference |Description |
|-|-|-|
|<b>[Tiny YOLOv2](vision/object_detection_segmentation/tiny_yolov2)</b>|[Redmon et al.](https://arxiv.org/pdf/1612.08242.pdf)|A real-time CNN for object detection that detects 20 different classes. A smaller version of the more complex full YOLOv2 network.|
|<b>[SSD](vision/object_detection_segmentation/ssd)</b>|[Liu et al.](https://arxiv.org/abs/1512.02325)|Single Stage Detector: real-time CNN for object detection that detects 80 different classes.|
|<b>[Faster-RCNN](vision/object_detection_segmentation/faster-rcnn)</b>|[Ren et al.](https://arxiv.org/abs/1506.01497)|Increases efficiency from R-CNN by connecting a RPN with a CNN to create a single, unified network for object detection that detects 80 different classes.|
|<b>[Mask-RCNN](vision/object_detection_segmentation/mask-rcnn)</b>|[He et al.](https://arxiv.org/abs/1703.06870)|A real-time neural network for object instance segmentation that detects 80 different classes. Extends Faster R-CNN as each of the 300 elected ROIs go through 3 parallel branches of the network: label prediction, bounding box prediction and mask prediction.|
|YOLO v2|[Redmon et al.](https://arxiv.org/abs/1612.08242)|A CNN model for real-time object detection system that can detect over 9000 object categories. It uses a single network evaluation, enabling it to be more than 1000x faster than R-CNN and 100x faster than Faster R-CNN. <br>[contribute](contribute.md)|
|<b>[YOLO v3](vision/object_detection_segmentation/yolov3)</b>|[Redmon et al.](https://pjreddie.com/media/files/papers/YOLOv3.pdf)|A deep CNN model for real-time object detection that detects 80 different classes. A little bigger than YOLOv2 but still very fast. As accurate as SSD but 3 times faster.|
|<b>[DUC](vision/object_detection_segmentation/duc)</b>|[Wang et al.](https://arxiv.org/abs/1702.08502)|Deep CNN based pixel-wise semantic segmentation model with >80% [mIOU](/models/semantic_segmentation/DUC/README.md/#metric) (mean Intersection Over Union). Trained on cityscapes dataset, which can be effectively implemented in self driving vehicle systems.|
|FCN|[Long et al.](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)|Deep CNN based segmentation model trained end-to-end, pixel-to-pixel that produces efficient inference and learning. Built off of AlexNet, VGG net, GoogLeNet classification methods. <br>[contribute](contribute.md)|
<hr>

### Body, Face & Gesture Analysis <a name="body_analysis"/>
Face detection models identify and/or recognize human faces and emotions in given images. Body and Gesture Analysis models identify gender and age in given image. 

|Model Class |Reference |Description |
|-|-|-|
|<b>[ArcFace](vision/body_analysis/arcface)</b>|[Deng et al.](https://arxiv.org/abs/1801.07698)|A CNN based model for face recognition which learns discriminative features of faces and produces embeddings for input face images.|
|CNN Cascade|[Li et al.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf)|The model operates at multiple resolutions, quickly rejecting the background regions in the fast low resolution stages in an image and carefully evaluates a small number of challenging candidates in the last high resolution stage. <br>[contribute](contribute.md)|
|<b>[Emotion FerPlus](vision/body_analysis/emotion_ferplus)</b> |[Barsoum et al.](https://arxiv.org/abs/1608.01041)	| Deep CNN for emotion recognition trained on images of faces.|
|Age and Gender Classification using Convolutional Neural Networks| [Levi et al.](https://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf)	|This model accurately classifies gender and age even the amount of learning data is limited.<br>[contribute](contribute.md)|
<hr>

### Image Manipulation <a name="image_manipulation"/>
Image manipulation models use neural networks to transform input images to modified output images. Some popular models in this category involve style transfer or enhancing images by increasing resolution.

|Model Class |Reference |Description |
|-|-|-|
|Unpaired Image to Image Translation using Cycle consistent Adversarial Network|[Zhu et al.](https://arxiv.org/abs/1703.10593)|The model uses learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. <br>[contribute](contribute.md)|
|<b>[Super Resolution with sub-pixel CNN](vision/super_resolution/sub_pixel_cnn_2016)</b> |	[Shi et al.](https://arxiv.org/abs/1609.05158)	|A deep CNN that uses sub-pixel convolution layers to upscale the input image. |
|<b>[Fast Neural Style Transfer](vision/style_transfer/fast_neural_style)</b> |	[Johnson et al.](https://arxiv.org/abs/1603.08155)	|This method uses a loss network pretrained for image classification to define perceptual loss functions that measure perceptual differences in content and style between images. The loss network remains fixed during the training process.|
<hr>

### Speech & Audio Processing <a name="speech"/>
This class of models uses audio data to train models that can identify voice, generate music, or even read text out loud.

|Model Class |Reference |Description |
|-|-|-|
|Speech recognition with deep recurrent neural networks|	[Graves et al.](https://www.cs.toronto.edu/~fritz/absps/RNN13.pdf)|A RNN model for sequential data for speech recognition. Labels problems where the input-output alignment is unknown<br>[contribute](contribute.md)|
|Deep voice: Real time neural text to speech |	[Arik et al.](https://arxiv.org/abs/1702.07825)	|A DNN model that performs end-to-end neural speech synthesis. Requires fewer parameters and it is faster than other systems. <br>[contribute](contribute.md)|
|Sound Generative models|	[WaveNet: A Generative Model for Raw Audio ](https://arxiv.org/abs/1609.03499)|A CNN model that generates raw audio waveforms. Has predictive distribution for each audio sample. Generates realistic music fragments. <br>[contribute](contribute.md)|
<hr>

### Machine Comprehension <a name="machine_comprehension"/>
This subset of natural language processing models that answer questions about a given context paragraph.

|Model Class |Reference |Description |
|-|-|-|
|<b>[Bidirectional Attention Flow](text/machine_comprehension/bidirectional_attention_flow)</b>|[Seo et al.](https://arxiv.org/pdf/1611.01603)|A model that answers a query about a given context paragraph.|
|<b>[BERT-Squad](text/machine_comprehension/bert-squad)</b>|[Devlin et al.](https://arxiv.org/pdf/1810.04805.pdf)|This model answers questions based on the context of the given input paragraph. |
<hr>

### Machine Translation <a name="machine_translation"/>
This class of natural language processing models learns how to translate input text to another language.

|Model Class |Reference |Description |
|-|-|-|
|Neural Machine Translation by jointly learning to align and translate|	[Bahdanau et al.](https://arxiv.org/abs/1409.0473)|Aims to build a single neural network that can be jointly tuned to maximize the translation performance. <br>[contribute](contribute.md)|
|Google's Neural Machine Translation System|	[Wu et al.](https://arxiv.org/abs/1609.08144)|This model helps to improve issues faced by the Neural Machine Translation (NMT) systems like parallelism that helps accelerate the final translation speed.<br>[contribute](contribute.md)|
<hr>

### Language Modelling <a name="language"/>
This subset of natural language processing models learns representations of language from large corpuses of text.

|Model Class |Reference |Description |
|-|-|-|
|Deep Neural Network Language Models | [Arisoy et al.](https://pdfs.semanticscholar.org/a177/45f1d7045636577bcd5d513620df5860e9e5.pdf)|A DNN acoustic model. Used in many natural language technologies. Represents a probability distribution over all possible word strings in a language. <br> [contribute](contribute.md)|
<hr>

### Visual Question Answering & Dialog <a name="visual_qna"/>
This subset of natural language processing models uses input images to answer questions about those images.

|Model Class |Reference |Description |
|-|-|-|
|VQA: Visual Question Answering |[Agrawal et al.](https://arxiv.org/pdf/1505.00468v6.pdf)|A model that takes an image and a free-form, open-ended natural language question about the image and outputs a natural-language answer. <br>[contribute](contribute.md)|
|Yin and Yang: Balancing and Answering Binary Visual Questions |[Zhang et al.](https://arxiv.org/pdf/1511.05099.pdf)|Addresses VQA by converting the question to a tuple that concisely summarizes the visual concept to be detected in the image. Next, if the concept can be found in the image, it provides a “yes” or “no” answer. Its performance matches the traditional VQA approach on unbalanced dataset, and outperforms it on the balanced dataset. <br>[contribute](contribute.md)|
|Making the V in VQA Matter|[Goyal et al.](https://arxiv.org/pdf/1612.00837.pdf)|Balances the VQA dataset by collecting complementary images such that every question is associated with a pair of similar images that result in two different answers to the question, providing a unique interpretable model that provides a counter-example based explanation.  <br>[contribute](contribute.md)|
|Visual Dialog|	[Das et al.](https://arxiv.org/abs/1611.08669)|An AI agent that holds a meaningful dialog with humans in natural, conversational language about visual content. Curates a large-scale Visual Dialog dataset (VisDial). <br>[contribute](contribute.md)|
<hr>

### Other interesting models <a name="others"/>
There are many interesting deep learning models that do not fit into the categories described above. The ONNX team would like to highly encourage users and researchers to [contribute](contribute.md) their models to the growing model zoo.

|Model Class |Reference |Description |
|-|-|-|
|Text to Image|	[Generative Adversarial Text to image Synthesis ](https://arxiv.org/abs/1605.05396)|Effectively bridges the advances in text and image modeling, translating visual concepts from characters to pixels. Generates plausible images of birds and flowers from detailed text descriptions. <br>[contribute](contribute.md)|
|Time Series Forecasting|	[Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks ](https://arxiv.org/pdf/1703.07015.pdf)|The model extracts short-term local dependency patterns among variables and to discover long-term patterns for time series trends. It helps to predict solar plant energy output, electricity consumption, and traffic jam situations. <br>[contribute](contribute.md)|
|Recommender systems|[DropoutNet: Addressing Cold Start in Recommender Systems](http://www.cs.toronto.edu/~mvolkovs/nips2017_deepcf.pdf)|A collaborative filtering method that makes predictions about an individual’s preference based on preference information from other users.<br>[contribute](contribute.md)|
|Collaborative filtering|[Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)|A DNN model based on the interaction between user and item features using matrix factorization. <br>[contribute](contribute.md)|
|Autoencoders|[A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://arxiv.org/abs/1506.01057)|An LSTM (long-short term memory) auto-encoder to preserve and reconstruct multi-sentence paragraphs.<br>[contribute](contribute.md)|
<hr>

## Usage <a name="usage-"/>

Every ONNX backend should support running the models out of the box. After downloading and extracting the tarball of each model, you will find:

- A protobuf file `model.onnx` that represents the serialized ONNX model.
- Test data (in the form of serialized protobuf TensorProto files or serialized NumPy archives).

The test data files can be used to validate ONNX models from the Model Zoo. We have provided the following interface examples for you to get started. Please replace `onnx_backend` in your code with the appropriate framework of your choice that provides ONNX inferencing support, and likewise replace `backend.run_model` with the framework's model evaluation logic. 

There are two different formats for the test data files:

- Serialized protobuf TensorProtos (.pb), stored in folders with the naming convention `test_data_set_*`.

```python
import numpy as np
import onnx
import os
import glob
import onnx_backend as backend

from onnx import numpy_helper

model = onnx.load('model.onnx')
test_data_dir = 'test_data_set_0'

# Load inputs
inputs = []
inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
for i in range(inputs_num):
    input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(input_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    inputs.append(numpy_helper.to_array(tensor))

# Load reference outputs
ref_outputs = []
ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
for i in range(ref_outputs_num):
    output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
    tensor = onnx.TensorProto()
    with open(output_file, 'rb') as f:
        tensor.ParseFromString(f.read())
    ref_outputs.append(numpy_helper.to_array(tensor))

# Run the model on the backend
outputs = list(backend.run_model(model, inputs))

# Compare the results with reference outputs.
for ref_o, o in zip(ref_outputs, outputs):
    np.testing.assert_almost_equal(ref_o, o)
```

- Serialized Numpy archives, stored in files with the naming convention `test_data_*.npz`. Each file contains one set of test inputs and outputs.

```python
import numpy as np
import onnx
import onnx_backend as backend

# Load the model and sample inputs and outputs
model = onnx.load(model_pb_path)
sample = np.load(npz_path, encoding='bytes')
inputs = list(sample['inputs'])
outputs = list(sample['outputs'])

# Run the model with an onnx backend and verify the results
np.testing.assert_almost_equal(outputs, backend.run_model(model, inputs))
```

## Model Visualization
You can see visualizations of each model's network architecture by using [Netron](https://lutzroeder.github.io/Netron).

## Contributions
Do you want to contribute a model? To get started, pick any model presented above with the [contribute](contribute.md) link under the Description column. The links point to a page containing guidelines for making a contribution.

# License

Apache License
                           Version 2.0, January 2004
                        https://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright 2019 Rolando Gopez Lacuata.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       https://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

