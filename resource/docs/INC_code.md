### Intel® Neural Compressor Code-based Demo

This is an example showing how to quantize an ONNX model with [Intel® Neural Compressor](https://github.com/intel/neural-compressor) step by step:

- Config file

```yaml
model:
  name: alexnet
  framework: onnxrt_qlinearops

quantization:
  approach: post_training_static_quant

evaluation:
  accuracy:
    metric:
      topk: 1

tuning:
  accuracy_criterion:
    relative: 0.01 # accuracy target
```

- Launcher code

```python
import numpy as np
import re
import os
from PIL import Image

# extract dataset class from inference code
class dataset:
    def __init__(self, data_path, image_list):
        self.image_list = []
        self.label_list = []
        with open(image_list, 'r') as f:
            for s in f:
                image_name, label = re.split(r"\s+", s.strip())
                src = os.path.join(data_path, image_name)
                if not os.path.exists(src):
                    continue
                self.image_list.append(src)
                self.label_list.append(int(label))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path, label = self.image_list[index], self.label_list[index]
        with Image.open(image_path) as image:
            image = np.array(image.convert('RGB').resize((224, 224))).astype(np.float32)
            image[:, :, 0] -= 123.68
            image[:, :, 1] -= 116.779
            image[:, :, 2] -= 103.939
            image[:,:,[0,1,2]] = image[:,:,[2,1,0]]
            image = image.transpose((2, 0, 1))
        return image, label

from neural_compressor.experimental import Quantization, common
ds = dataset('/path/to/imagenet', '/path/to/label')
quantize = Quantization('/path/to/config_file')
quantize.calib_dataloader = common.DataLoader(ds)
quantize.model = model
q_model = quantize()
q_model.save("int8.onnx")
```
