# Python-API-Script
Onnx Python API Script used to download and save pretrained models from [onnx model zoo](https://github.com/onnx/models). Retrieves metadata after the model is successfully downloaded. 

## Features 


```get_model_versions(MODEL_FOLDER_NAME)``` - Retrieves an array of all the versions of the specificed model folder in the Onnx Model Zoo. 
   
```get_pretrained()``` - Downloads and saves specific onnx models to desired path.
            

```get_metadata()``` - Retrieves metadata of the onnx model. 
        
## Intialization and Usage
Initiate the object by calling onnx_zoo class name.
```
OBJECT_NAME = onnx_zoo()
```
The Python scipt will then ask to input the model folder name. When inputting the name, it should be in all lowercase. 

```
Enter Model Name: resnet
```
                       
After model folder name is inputted, the script will output all the model versions that exist in the folder. 

```
['resnet101-v1-7', 'resnet101-v2-7', 'resnet152-v1-7', 'resnet152-v2-7', 'resnet18-v1-7', 'resnet18-v2-7', 'resnet34-v1-7', 'resnet34-v2-7', 'resnet50-caffe2-v1-3', 'resnet50-caffe2-v1-6', 'resnet50-caffe2-v1-7', 'resnet50-caffe2-v1-8', 'resnet50-caffe2-v1-9', 'resnet50-v1-7', 'resnet50-v2-7']
```

From the array of versions, input the version that will be downloaded and input the local directory that the model will be saved in.

``` 
Enter model name from options: resnet101-v1-7 
Enter saved path: /Users/name/Downloads
```

To download the model and output its metadata, run the following functions:

``` 
OBJECT_NAME.get_pretrained()
OBJECT_NAME.get_metadata()
```

## Installation 
Install onnx to check models

```pip install onnx```

Install [onnxruntime](https://github.com/microsoft/onnxruntime) to run onnx models

```pip install onnxruntime```
