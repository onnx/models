# BiDAF

 ## Description
This model is a neural network for answering a query about a given context paragraph.

 ## Model

 |Model        |Download  |Checksum|Download (with sample test data)|ONNX version|Opset version|Accuracy |
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|BiDAF  |[41.5 MB](bidaf.onnx) | [MD5](bidaf-md5.txt) |[37.3 MB](bidaf.tar.gz)|1.4 |ONNX 9, ONNX.ML 1 |EM of 68.1 in SQuAD v1.1 |

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

 ## Publication/Attribution
Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi. Bidirectional Attention Flow for Machine Comprehension, [paper](https://arxiv.org/abs/1611.01603)

 <hr>

 ## References
This model is converted from a CNTK model trained from [this implementation](https://github.com/microsoft/CNTK/tree/nikosk/bidaf/Examples/Text/BidirectionalAttentionFlow/squad).
<hr>

 ## License
MIT License
<hr>
