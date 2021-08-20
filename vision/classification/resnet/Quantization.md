<!--- SPDX-License-Identifier: Apache-2.0 -->

# ResNet

## Inference
We use onnxruntime to perform inference. View the notebook [onnxrt_inference](../../onnxrt_inference.ipynb) to understand how to use above models for doing inference.

### Input
The fp32 and int8 models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (N x 3 x H x W), where N is the batch size expected to be 1, and H and W are expected to be 224.

### Preprocessing
```python
def preprocess(img):
    img = img / 255.
    img = cv2.resize(img, (256, 256))
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img
```

### Output
The model outputs image scores for each class.

### Postprocessing
The post-processing sort the predictions to report the most probable classes.

```python
preds = np.squeeze(preds)
a = np.argsort(preds)[::-1]
print('class=%s ; probability=%f' %(labels[a[0]],preds[a[0]]))
```

### Dataset
Please refer to [Resnet Description](../README.md) for dataset information.

## Quantization
ResNet50_int8 is obtained by quantizing ResNet50_fp32 model. We use [Intel® Low Precision Optimization Tool (LPOT)](https://github.com/intel/lpot) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel/lpot/tree/master/examples/onnxrt/onnx_model_zoo/resnet50/README.md) to understand how to use LPOT for quantization.

### Environment
onnx: 1.7.0 
onnxruntime: 1.6.0+

### Prepare model
```shell
wget https://github.com/onnx/models/raw/master/vision/classification/resnet/model_quantized/model/resnet50-v1-12.onnx
```

### Model quantize
Make sure to specify the appropriate dataset path in the configuration file.
```bash
bash run_tuning.sh --input_model=path/to/model \  # model path as *.onnx
                   --config=resnet50_v1_5.yaml \
                   --output_model=path/to/save
```

### References
* **ResNetv1**
[Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385)
 He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.

* Intel® Low Precision Optimization Tool (LPOT): https://github.com/intel/lpot

## Contributors
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)

## License
Apache 2.0
