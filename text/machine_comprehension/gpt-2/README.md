# GPT-2

## Use-cases
Transformer-based language model for text generation.

## Description
[GPT-2](https://openai.com/blog/better-language-models/) is a large transformer-based language model with a simple objective: predict the next word, given all of the previous words within some text.

## Model

 |Model        |Download  |Checksum| Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|GPT-2       |[522.81 MB](https://onnxzoo.blob.core.windows.net/models/opset_10/GPT2/model.onnx) | [MD5](https://onnxzoo.blob.core.windows.net/models/opset_10/GPT2/gpt2-md5.txt)| [438.3 MB](https://onnxzoo.blob.core.windows.net/models/opset_10/GPT2/GPT2.tar.xz)| 1.6 | 10 |mAP of [0.024](https://docs.google.com/spreadsheets/d/1sryqufw2D0XlUH4sq3e9Wnxu5EAQkaohzrJbd5HdQ_w/edit#gid=0)|


## Inference

### Input to model
Sequence of words as a string. Example: "Here is some text to encode : Hello World", tokenized by Byte-Pair-Encoding.
**input_ids**: Indices of input tokens in the vocabulary.
**past**: precomputed hidden-states
Other optional inputs to the model are:
**attention_mask**: Mask tensor that has a value of 1 for real input tokens and 0 for padding tokens.
**token_type_ids**: A parallel sequence of tokens (can be used to indicate various portions of the inputs).
**position_ids**: Indices of tokens in the position embeddings.
**head_mask**: Mask tensor that has a value of 1 when head is not maked, and 0 when head is masked.

### Preprocessing steps
Use ```tokenizer.encode()``` to encode the input text:
```python
text = "Here is some text to encode : Hello World"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens_tensor = torch.tensor([torch.tensor(tokenizer.encode(text))])
```

### Output of model
Output tuple can have different sizes depending on the configurations and inputs.
**last_hidden_state**: Sequence of hidden-states at the last layer of the model.
**past**: pre-computed hidden-states.
Optional outputs are:
**hidden_states**: Hidden-states of the model at the output of each layer
**attentions**: Attentions weights

### Postprocessing steps
```python
outputs = model(input_ids)
last_hidden_states = outputs[0]
```
<hr>

## Dataset (Train and validation)
The original model from OpenAI is pretrained on a dataset of [8 million web pages](https://openai.com/blog/better-language-models).
The pretrained model is referenced in  [huggingface/transformers](https://github.com/huggingface/transformers/blob/master/transformers/modeling_gpt2.py) repository as a causal (unidirectional) transformer pre-trained using language modeling on a very large corpus of ~40 GB of text data.
https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin

<hr>

## Validation accuracy
Metric and benchmarking details are provided by Huggigngface in this [post](https://medium.com/huggingface/benchmarking-transformers-pytorch-and-tensorflow-e2917fb891c2).
<hr>


## Publication/Attribution
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, andIlya Sutskever. Language Models are Unsupervised Multitask Learners. 2019.

## References
This model is converted directly from [huggingface/transformers](https://github.com/huggingface/transformers)
<hr>

## Contributors
Negin Raoof
<hr>

## License
Add license information
<hr>
