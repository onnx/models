<!--- SPDX-License-Identifier: Apache-2.0 -->
# GPT-2 with Beam Search Generation

## Use-cases
Transformer-based language model for text generation.

## Description
This GPT-2 model with generation can produce the result without any extra code or algorithm. It already embedded a beam search algorithm into the ONNX model, so there is **NO** Post-Processing code to inference.


## Model

 |Model        |Download  | Compressed |ONNX version|Opset version|
|-------------|:--------------|:--------------|:--------------|:--------------|
|gpt2-lm-head-bs |[635 MB](model/gpt2-lm-head-bs-12.onnx) | N/A (similiar size) | 1.7 | 12


### Source
Huggingface PyTorch GPT-2-with-lm-head + conversion script ==> ONNX GPT-2-LM-HEAD-BS.onnx
The full conversion script is in [onnxruntime-extentions](https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/gpt2bs.py), and some model parameters can be changed if the number in the script was changed.

## Inference
running this model is straightforward, with onnxruntime-extensions, it only contains several lines for an end-to-end inference.
```python

from onnxruntime_extensions import PyOrtFunction

gpt2_all = PyOrtFunction.from_model('model/gpt2-lm-head-bs-12.onnx')
encdict = tokenizer('What is the best story', padding=True, return_tensors='np')

outputs = gpt2_all(encdict['input_ids'], encdict['attention_mask'].astype('float32'), 30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

and the tokenizer used in the above code example can be get like

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
```


## Publication/Attribution
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, andIlya Sutskever. Language Models are Unsupervised Multitask Learners. 2019.

## References
This model is converted directly from [huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_gpt2.py).
<hr>

## Contributors
Wenbing Li
<hr>

## License
Apache 2.0 License
<hr>