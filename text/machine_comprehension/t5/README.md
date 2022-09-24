<!--- SPDX-License-Identifier: Apache-2.0 -->

# T5

## Use-cases
Transformer-based language model trained on multiple tasks including summarization, sentiment-analysis, q&a, translation etc.
The implementation in this repo is an adaptation of the [onnxt5 repo](https://github.com/abelriboulot/onnxt5) which makes the export and use of T5 with ONNX easier.

## Description
[T5](https://arxiv.org/abs/1910.10683) is a transformer model which aims to provide great flexibility and provide better semantic
understanding through the training of multiple tasks at once.

## Model

 |Model        |Download  | Compressed |ONNX version|Opset version|
|-------------|:--------------|:--------------|:--------------|:--------------|
|T5-encoder       |[650.6 MB](model/t5-encoder-12.onnx) | [205.0 MB](model/t5-encoder-12.tar.gz)| 1.7 | 12
|T5-decoder-with-lm-head |[304.9 MB](model/t5-decoder-with-lm-head-12.onnx) | [304.9 MB](model/t5-decoder-with-lm-head-12.tar.gz)| 1.7 | 12


### Source
Huggingface PyTorch T5 + script changes ==> ONNX T5-encoder

Huggingface PyTorch T5 + script changes ==> ONNX T5-decoder-with-lm-head

Script changes include:
 - reshaping the Huggingface models to combine the lm head with the decoder to allow for a unified model
 - reshaping the encoder to output the hidden state directly

## Inference
The script for ONNX model conversion and ONNX Runtime inference is [here](dependencies/T5-export.py).
More complete utilities to export and use the models are maintained in the [onnxt5 repo](https://github.com/abelriboulot/onnxt5).

### Input to model
This implementation takes as inputs a prompt which begins by the task at hand here. Examples of some tasks include ```summarize: <PROMPT>```,
```translate English to French: <PROMPT>```, ```cola sentence: <PROMPT>```, etc.
For the full list of task you can refer to the appendix D of the [original paper](https://arxiv.org/pdf/1910.10683.pdf).


### Preprocessing steps
The easiest way to use the model is to use the onnxt5 utilities (installation instructions: ```pip install onnxt5```).

In that case you can use the model with the following piece of code:
```python
from onnxt5 import GenerativeT5
from onnxt5.api import get_encoder_decoder_tokenizer
decoder_sess, encoder_sess, tokenizer = get_encoder_decoder_tokenizer()
generative_t5 = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)
prompt = 'translate English to French: I was a victim of a series of accidents.'
output_text, output_logits = generative_t5(prompt, max_length=100, temperature=0.)
# output_text: "J'ai été victime d'une série d'accidents."
```

Or if you wish to produce the embeddings of a sentence:
```python
from onnxt5.api import get_encoder_decoder_tokenizer, run_embeddings_text

decoder_sess, encoder_sess, tokenizer = get_encoder_decoder_tokenizer()
prompt = 'Listen, Billy Pilgrim has come unstuck in time.'
encoder_embeddings, decoder_embeddings = run_embeddings_text(encoder_sess, decoder_sess, tokenizer, prompt)
```

Otherwise you can manually create the Generative model with the following:

```python
from onnxruntime import InferenceSession
from transformers import T5Tokenizer
from .dependencies.models import GenerativeT5

tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
# other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
# based on the build flags) when instantiating InferenceSession.
# For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
# InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])
decoder_sess = InferenceSession(str(path_t5_decoder))
encoder_sess = InferenceSession(str(path_t5_encoder))
generative_t5 = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)
generative_t5('translate English to French: I was a victim of a series of accidents.', 21, temperature=0.)[0]
```

### Output of model
For the T5-encoder model:

**last_hidden_state**: Sequence of hidden-states at the last layer of the model. It's a float tensor of size (batch_size, sequence_length, hidden_size).

For T5-decoder-with-lm-head model:

**logit_predictions**: Prediction scores of the language modeling head. It's a float tensor of size (batch_size, sequence_length, vocab_size).

### Postprocessing steps
For the T5-encoder model:

```python
last_hidden_states = model(input_ids)[0]
```

For the T5-decoder-with-lm-head model:

```python
# To generate the encoder's last hidden state
encoder_output = encoder_sess.run(None, {"input_ids": input_ids})[0]
# To generate the full model's embeddings
decoder_output = decoder_sess.run(None, {
                                        "input_ids": input_ids,
                                        "encoder_hidden_states": encoder_output
    })[0]
```

For the generative model, to generate a translation:
```
from onnxt5 import GenerativeT5
from onnxt5.api import get_encoder_decoder_tokenizer
decoder_sess, encoder_sess, tokenizer = get_encoder_decoder_tokenizer()
generative_t5 = GenerativeT5(encoder_sess, decoder_sess, tokenizer, onnx=True)
prompt = 'translate English to French: I was a victim of a series of accidents.'
output_text, output_logits = generative_t5(prompt, max_length=100, temperature=0.)
```
<hr>

## Dataset (Train and validation)
The original model from Google Brain is pretrained on the [Colossal Clean Crawled Corpus](https://www.tensorflow.org/datasets/catalog/c4).
The pretrained model is referenced in [huggingface/transformers](https://github.com/huggingface/transformers/blob/master/transformers/modeling_t5.py), trained on the same data.
<hr>

## Validation accuracy
Benchmarking can be run with the following [script](https://github.com/abelriboulot/onnxt5/blob/master/notebooks/benchmark_performance.ipynb) with initial results in this [post](https://kta.io/posts/onnx_t5).
<hr>


## Publication/Attribution
This repo is based on the work of Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and
Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu from Google, as well as the implementation of T5 from the
huggingface team, the work of the Microsoft ONNX and onnxruntime teams, in particular Tianlei Wu, and the work of Thomas Wolf on generation of text.

[Original T5 Paper](https://arxiv.org/pdf/1910.10683.pdf)
```
@article{2019t5,
  author = {Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu},
  title = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  journal = {arXiv e-prints},
  year = {2019},
  archivePrefix = {arXiv},
  eprint = {1910.10683},
}
```

## References
This model is converted directly from [huggingface/transformers](https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_t5.py).
<hr>

## Contributors
[Abel Riboulot](https://github.com/abelriboulot)
<hr>

## License
Apache 2.0 License
<hr>
