# ArcFace

## Use cases
For each face image, the model produces a fixed length embedding vector corresponding to the face in the image. The vectors from face images of a single person have a higher similarity than that from different persons. Therefore, the model is primarily used for face recognition/verification. It can also be used in other applications like facial feature based clustering.

## Description
ArcFace is a CNN based model for face recognition which learns discriminative features of faces and produces embeddings for input face images. To enhance the discriminative power of softmax loss, a novel supervisor signal called additive angular margin (ArcFace) is used here as an additive term in the softmax loss. ArcFace can use a variety of CNN networks as its backend, each having different accuracy and performance. 

## Model
The model LResNet100E-IR is an ArcFace model that uses ResNet100 as a backend with modified input and output layers.

|Model        |Download  |Download (with sample test data)|ONNX version|Opset version|LFW * accuracy (%)|CFP-FF * accuracy (%)|CFP-FP * accuracy (%)|AgeDB-30 * accuracy (%)|
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|LResNet100E-IR|    [248.9 MB](model/arcfaceresnet100-opset8.onnx)|[226.6 MB](model/arcfaceresnet100-opset8.tar.gz) | 1.3  |8|99.77     | 99.83  |  94.21     | 97.87|

\* each of the accuracy metrics correspond to accuracies on different [validation sets](#val_data) each with their own [validation methods](#val_method).

## Inference
We used MXNet as framework to perform inference. View the notebook [arcface_inference](arcface_inference.ipynb) to understand how to use above models for doing inference. A brief description of the inference process is provided below:

### Input 
The input to the model should preferably be images containing a single face in each image. There are no constraints on the size of the image. The example displayed in the inference notebook was done using jpeg images.

### Preprocessing
In order to input only face pixels into the network, all input images are passed through a pretrained face detection and alignment model, [MTCNN detector](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). The output of this model are landmark points and a bounding box corresponding to the face in the image. Using this output, the image is processed using affine transforms to generate the aligned face images which are input to the network. Check [face_preprocess.py](face_preprocess.py) and [inference notebook](arcface_inference.ipynb) for code.

### Output
The model outputs an embedding vector for the input face images. The size of the vector is tunable (512 for LResNet100E-IR).

### Postprocessing
The post-processing involves normalizing the output embedding vectors to have unit length. Check [face_postprocess.py](face_postprocess.py) for code.

To do quick inference with the model, check out [Model Server](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md/#arcface-resnet100_onnx).

## Dataset
### Training
**Refined MS-Celeb-1M** is a refined version of the [MS-Celeb-1M dataset](https://arxiv.org/abs/1607.08221). The refined version contains 3.8 million images from 85000 unique identities.

### <a name="val_data"></a>Validation
The following three datasets are used for validation:
* **Labelled Faces in the Wild (LFW)** contains 13233 web-collected images from 5749 identities with large variations in pose, exposure and illuminations.

* **Celebrities in Frontal Profile (CFP)** consists of 500 subjects, each with 10 frontal and 4 profile images.

* **Age Database (AgeDB)** is a dataset with large variations in pose, expression, illuminations, and age. AgeDB contains 12240 images of 440 distinct subjects, such as actors, actresses, writers, scientists, and politicians.  Each image is annotated with respect to the identity, age and gender attribute. The minimum and maximum ages are 3 and 101, respectively. The average age range for each subject is 49 years. There are four groups of test data with different year gaps (5, 10, 20 and 30 years respectively). The last subset, AgeDB-30 is used here as its the most challenging.

### Setup
1. Download the file  `faces_ms1m_112x112.zip` : [8.1 GB](https://s3.amazonaws.com/onnx-model-zoo/arcface/dataset/faces_ms1m_112x112.zip)
2. Unzip `faces_ms1m_112x112.zip` to produce a folder of the same name. Use path to this folder in the notebooks. This folder contains the training as well as the validation datasets.

## Validation accuracy
The accuracies obtained by the models on the validation set are mentioned above. Maximum deviation of 0.2%(CFP-FP) in accuracy is observed compared to that in the paper.

## Training
We used MXNet as framework to perform training. View the [training notebook](train_arcface.ipynb) to understand details for parameters and network for each of the above variants of ArcFace.

## <a name="val_method"></a>Validation
The validation techniques for the three validation sets are described below:
* **LFW** : Face verification accuracy on 6000 face pairs.

* **CFP** : Two types of face verification - Frontal-Frontal (FF) and Frontal-Profile (FP), each having 10 folders with 350 same-person pairs and 350 different person pairs.

* **AgeDB** : Validation is only performed on AgeDB-30 as mentioned above, with a metric same as LFW.

We used MXNet as framework to perform validation. Use the notebook [arcface_validation](arcface_validation.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook.

## References
* All models are from the paper [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698).
* Original training dataset from the paper [MS-Celeb-1M: A Dataset and Benchmark for Large-Scale Face Recognition](https://arxiv.org/abs/1607.08221).
* [InsightFace repo](https://github.com/deepinsight/insightface), [MXNet](http://mxnet.incubator.apache.org)

## Contributors
[abhinavs95](https://github.com/abhinavs95) (Amazon AI)

## License
Apache 2.0
