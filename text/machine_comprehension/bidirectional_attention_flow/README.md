<!--- SPDX-License-Identifier: MIT -->

# BiDAF

 ## Description
This model is a neural network for answering a query about a given context paragraph.

 ## Model

 |Model        |Download  |Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|BiDAF  |[41.5 MB](model/bidaf-9.onnx) |[37.3 MB](model/bidaf-9.tar.gz)|1.4 |ONNX 9, ONNX.ML 1 |EM of 68.1 in SQuAD v1.1 |
|BiDAF-int8  |[12 MB](model/bidaf-11-int8.onnx) |[8.7 MB](model/bidaf-11-int8.tar.gz)|1.13.1 |ONNX 11, ONNX.ML 1 |EM of 65.93 in SQuAD v1.1 |
> Compared with the fp32 BiDAF, int8 BiDAF accuracy drop ratio is 0.23% and performance improvement is 0.89x in SQuAD v1.1.
>
> The performance depends on the test hardware. Performance data here is collected with Intel® Xeon® Platinum 8280 Processor, 1s 4c per instance, CentOS Linux 8.3, data batch size is 1.

 <hr>

 ## Inference

 ### Input to model
 Tokenized strings of context paragraph and query.

 ### Preprocessing steps
 Tokenize words and chars in string for context and query. The tokenized words are in lower case, while chars are not. Chars of each word needs to be clamped or padded to list of length 16. Note [NLTK](https://www.nltk.org/install.html) is used in preprocess for word tokenize.

* context_word: [seq, 1,] of string
* context_char: [seq, 1, 1, 16] of string
* query_word: [seq, 1,] of string
* query_char: [seq, 1, 1, 16] of string

 The following code shows how to preprocess input strings:

 ```python
import numpy as np
import string
from nltk import word_tokenize

def preprocess(text):
    tokens = word_tokenize(text)
    # split into lower-case word tokens, in numpy array with shape of (seq, 1)
    words = np.asarray([w.lower() for w in tokens]).reshape(-1, 1)
    # split words into chars, in numpy array with shape of (seq, 1, 1, 16)
    chars = [[c for c in t][:16] for t in tokens]
    chars = [cs+['']*(16-len(cs)) for cs in chars]
    chars = np.asarray(chars).reshape(-1, 1, 1, 16)
    return words, chars

# input
context = 'A quick brown fox jumps over the lazy dog.'
query = 'What color is the fox?'
cw, cc = preprocess(context)
qw, qc = preprocess(query)
```

 ### Output of model
The model has 2 outputs.

* start_pos: the answer's start position (0-indexed) in context,
* end_pos: the answer's inclusive end position (0-indexed) in context.

 ### Postprocessing steps
Post processing and meaning of output
```
# assuming answer contains the np arrays for start_pos/end_pos
start = np.asscalar(answer[0])
end = np.asscalar(answer[1])
print([w.encode() for w in cw[start:end+1].reshape(-1)])
```

For this testcase, it would output
```
[b'brown'].
```
<hr>

 ## Dataset (Train and validation)
The model is trained with [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/).
<hr>

 ## Validation accuracy
Metric is Exact Matching (EM) of 68.1, computed over SQuAD v1.1 dev data.
<hr>

## Quantization
BiDAF-int8 is obtained by quantizing fp32 BiDAF model. We use [Intel® Neural Compressor](https://github.com/intel/neural-compressor) with onnxruntime backend to perform quantization. View the [instructions](https://github.com/intel/neural-compressor/blob/master/examples/onnxrt/nlp/onnx_model_zoo/BiDAF/quantization/ptq_dynamic/README.md) to understand how to use Intel® Neural Compressor for quantization.


### Prepare Model
Download model from [ONNX Model Zoo](https://github.com/onnx/models).

```shell
wget https://github.com/onnx/models/raw/main/text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.onnx
```

Convert opset version to 11 for more quantization capability.

```python
import onnx
from onnx import version_converter

model = onnx.load('bidaf-9.onnx')
model = version_converter.convert_version(model, 11)
onnx.save_model(model, 'bidaf-11.onnx')
```

### Model quantize

Dynamic quantization:

```bash
bash run_tuning.sh --input_model=path/to/model \ # model path as *.onnx
                   --dataset_location=path/to/squad/dev-v1.1.json
                   --output_model=path/to/model_tune
```

<hr>

 ## Publication/Attribution
Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi. Bidirectional Attention Flow for Machine Comprehension, [paper](https://arxiv.org/abs/1611.01603)

 <hr>

 ## References
* This model is converted from a CNTK model trained from [this implementation](https://github.com/microsoft/CNTK/tree/nikosk/bidaf/Examples/Text/BidirectionalAttentionFlow/squad).
* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)
<hr>

## Contributors
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [yuwenzho](https://github.com/yuwenzho) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)
<hr>

 ## License
MIT License
<hr>
