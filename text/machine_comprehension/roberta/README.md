# RoBERTa

## Use cases
This transformer-based sequence clssification model that predicts sentiment based on given input text.   

## Description
RoBERTa builds on BERT’s language masking strategy and modifies key hyperparameters in BERT, including removing BERT’s next-sentence pretraining objective, and training with much larger mini-batches and learning rates. RoBERTa was also trained on an order of magnitude more data than BERT, for a longer amount of time. This allows RoBERTa representations to generalize even better to downstream tasks compared to BERT.

## Model

 |Model        |Download  |Download (with sample test data)| ONNX version |Opset version|Accuracy|
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|RoBERTa| [249 MB](model/roberta-9.onnx) |  [231 MB](model/roberta-9.tar.gz) |  1.4 | 9|mAP of [0.0183](https://docs.google.com/spreadsheets/d/1sryqufw2D0XlUH4sq3e9Wnxu5EAQkaohzrJbd5HdQ_w/edit#gid=0)|

## Source
PyTorch RoBERTa => ONNX RoBERTa

## Conversion
Tutorial for conversion can be found in the [conversion](https://github.com/SeldonIO/seldon-models/blob/master/pytorch/moviesentiment_roberta/pytorch-roberta-onnx.ipynb) notebook

## Inference
We used [ONNX Runtime](https://github.com/microsoft/onnxruntime) to perform the inference.

Tutorial for running inference using onnxruntime can be found in the [inference](dependencies/roberta-inference.ipynb) notebook.

### Input
Input is a sequence of words as a string. Example: "Text to encode : Hello World", tokenized by RobertaTokenizer. input_ids: Indices of input tokens in the vocabulary. It's a int64 tensor of dynamic shape (batch_size, sentence_length).

### Preprocessing
Use tokenizer.encode() to encode the input text:
```python
text = "This film is had bad graphics but overall it is good"
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
```

### Output
Output of this model is the tuple (batch_size, 2)

### Postprocessing
Print sentiment prediction
```python
pred = np.argmax(ort_out)
if(pred == 0):
    print("Prediction: negative")
elif(pred == 1):
    print("Prediction: positive")
```

## Dataset
Pretrained roberta weights can be downloaded [here](https://storage.googleapis.com/seldon-models/pytorch/moviesentiment_roberta/pytorch_model.bin).

## Validation accuracy
Metric and benchmarking details are provided by HuggingFace in this [post](https://medium.com/huggingface/benchmarking-transformers-pytorch-and-tensorflow-e2917fb891c2)

## Publication/Attribution
* [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf).Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov

## References
This model is converted directly from [seldon-models/pytorch](https://github.com/SeldonIO/seldon-models/blob/master/pytorch/moviesentiment_roberta/pytorch-roberta-onnx.ipynb)

## Contributors
[Kundana Pillari](https://github.com/kundanapillari)

## License
Apache 2.0