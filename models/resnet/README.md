# Resnet
## ONNX model
* Version 1:   
 Resnet-18 ; Size: 5MB  
 Resnet-34 ; Size: 5MB  
 Resnet-50 ; Size: 5MB  
 Resnet-101 ; Size: 5MB  
 Resnet-152 ; Size: 5MB  
 
* Version 2:  
  Resnet-18 ; Size: 5MB  
 Resnet-34 ; Size: 5MB  
 Resnet-50 ; Size: 5MB  
 Resnet-101 ; Size: 5MB  
 Resnet-152 ; Size: 5MB


## MMS archive
* Version 1:   
 Resnet-18 ; Size: 5MB  
 Resnet-34 ; Size: 5MB  
 Resnet-50 ; Size: 5MB  
 Resnet-101 ; Size: 5MB  
 Resnet-152 ; Size: 5MB  
 
* Version 2:  
  Resnet-18 ; Size: 5MB  
 Resnet-34 ; Size: 5MB  
 Resnet-50 ; Size: 5MB  
 Resnet-101 ; Size: 5MB  
 Resnet-152 ; Size: 5MB
## Dataset
Dataset used for train and validation: [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/). 

Check [imagenet_prep](../imagenet_prep.md) for guidelines on preparing the dataset. 
## Example notebook
View the notebook [imagenet_inference](../imagenet_inference.ipynb) to understand how to use above models for doing inference. Make sure to specify the appropriate model name in the notebook

## Training notebook
View the notebook that can be used for training [here](train_resnet.ipynb). The notebook contains details for 
parameters and network for each of the above variants of Resnet.
## Verify validation accuracy
Use the notebook [imagenet_verify](../imagenet_verify.ipynb) to verify the accuracy of the model on the validation set. Make sure to specify the appropriate model name in the notebook
## References
**Resnet-v1**  
[Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385)  
 He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.  
  **Resnet-v2**  
[Identity mappings in deep residual networks](https://arxiv.org/abs/1603.05027)  
He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
In European Conference on Computer Vision, pp. 630-645. Springer, Cham, 2016.

 
## Accuracy measures
## Tags

