# GPT-2

## Use-cases
Transformer-based language model for text generation.

## Description
[GPT-2](https://openai.com/blog/better-language-models/) is a large transformer-based language model with a simple objective: predict the next word, given all of the previous words within some text.

## Model

 |Model        |Download  | Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|GPT-2       |[522.81 MB](model/gpt2-10.onnx) | [438.3 MB](model/gpt2-10.tar.gz)| 1.6 | 10 |mAP of [0.024](https://docs.google.com/spreadsheets/d/1sryqufw2D0XlUH4sq3e9Wnxu5EAQkaohzrJbd5HdQ_w/edit#gid=0)|
|GPT-2-LM-HEAD |[664.87 MB](model/gpt2-lm-head-10.onnx) | [607 MB](model/gpt2-lm-head-10.tar.gz)| 1.6 | 10 |mAP of [0.024](https://docs.google.com/spreadsheets/d/1sryqufw2D0XlUH4sq3e9Wnxu5EAQkaohzrJbd5HdQ_w/edit#gid=0)|


### Source
PyTorch GPT-2 ==> ONNX GPT-2  
PyTorch GPT-2 + script changes ==> ONNX GPT-2-LM-HEAD


## Inference
The script for ONNX model conversion and ONNX Runtime inference is [here](dependencies/GPT2-export.py).

### Input to model
Sequence of words as a string. Example: "Here is some text to encode : Hello World", tokenized by Byte-Pair-Encoding.
**input_ids**: Indices of input tokens in the vocabulary. It's a long tensor of dynamic shape (batch_size, sequence_length).



### Preprocessing steps
Use ```tokenizer.encode()``` to encode the input text:
```python
text = "Here is some text to encode : Hello World"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens_tensor = torch.tensor([torch.tensor(tokenizer.encode(text))])
```

### Output of model
For GPT-2 model:

**last_hidden_state**: Sequence of hidden-states at the last layer of the model. It's a float tensor of size (batch_size, sequence_length, hidden_size).
**past**: pre-computed hidden-states. It's a list of tensors (key and values in the attention blocks) of size (batch_size, num_heads, sequence_length, sequence_length), one per each layer.

Output of this model is the tuple (last_hidden_state, past)

For GPT-2-LM-HEAD model:

**prediction_scores**: Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax). It's a float tensor of size (batch_size, sequence_length, vocab_size).
**past**: pre-computed hidden-states. It's a list of tensors (key and values in the attention blocks) of size (batch_size, num_heads, sequence_length, sequence_length), one per each layer.

Output of this model is the tuple (prediction_scores, past)

Note that output_hidden_states=False and output_attentions=False in the PretrainedConfig configs.

### Postprocessing steps
For GPT-2 model:

```python
outputs = model(input_ids)
last_hidden_states = outputs[0]
```

For GPT-2-LM-HEAD model, to generate next 10 words:
```
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

batch_size = 1
length = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Here is some text to encode : Hello World!"
tokens = np.array(tokenizer.encode(text))
context = torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
prev = context
output = context

for i in range(length):
    outputs = model(prev)
    logits = outputs[0]
    logits = logits[:, -1, :]
    log_probs = F.softmax(logits, dim=-1)
    _, prev = torch.topk(log_probs, k=1, dim=-1)
    output = torch.cat((output, prev), dim=1)

output = output[:, len(tokens):].tolist()
generated = 0
for i in range(batch_size):
    generated += 1
    text = tokenizer.decode(output[i])
    print(text)
```
<hr>

## Dataset (Train and validation)
The original model from OpenAI is pretrained on a dataset of [8 million web pages](https://openai.com/blog/better-language-models).
The pretrained model is referenced in  [huggingface/transformers](https://github.com/huggingface/transformers/blob/master/transformers/modeling_gpt2.py) repository as a causal (unidirectional) transformer pre-trained using language modeling on a very large corpus of ~40 GB of text data.
https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin

<hr>

## Validation accuracy
Metric and benchmarking details are provided by HuggingFace in this [post](https://medium.com/huggingface/benchmarking-transformers-pytorch-and-tensorflow-e2917fb891c2).
<hr>


## Publication/Attribution
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, andIlya Sutskever. Language Models are Unsupervised Multitask Learners. 2019.

## References
This model is converted directly from [huggingface/transformers](https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py).
<hr>

## Contributors
Negin Raoof  
Joddiy Zhang
<hr>

## License
Apache 2.0 License
<hr>
