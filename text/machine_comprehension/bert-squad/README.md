# BERT-Squad

## Use cases
This model uses masked language method to enable training of bidirectional models that can quickly and efficiently run through language processing tasks. The model also adds a next sentence prediction task to improve sentence-level understanding which helps to answer questions based on the given paragraph input. 

## Description
BERT (Bidirectional Encoder Representations from Transformers) applies Transformers, a popular attention model, to language modelling. This mechanism has an encoder to read the input text and a decoder that produces a prediction for the task. This model uses the technique of masking out some of the words in the input and then condition each word bidirectionally to predict the masked words. BERT also learns to model relationships between sentences, predicts if the sentences are connected or not. 

## Model

 |Model        |Download  |Checksum|Download (with sample test data)| ONNX version |Opset version|Dependencies| 
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|BERT-Squad| [393,930 KB](model/bert.onnx)  |dc59794145b4a37995a2667370dc3d6f |  [78 KB](model/saved_model.pb.gz) |  1.5.2  | 8| [tokenization.py](dependencies/tokenization.py) [run_onnx_squad.py](dependencies/run_onnx_squad.py)  | 

## Inference
We used onnxruntime to preform the inference. 

### Input 
Context paragraph and questions about the paragraph that are written in an inputs.json file. 
The following code shows to write the input file.
```python
%%writefile inputs.json
{
  "version": "1.5.2",
  "data": [
    {
      "paragraphs": [
        {
          "context": "In its early years, the new convention center failed to meet attendance and revenue expectations.[12] By 2002, many Silicon Valley businesses were choosing the much larger Moscone Center in San Francisco over the San Jose Convention Center due to the latter's limited space. A ballot measure to finance an expansion via a hotel tax failed to reach the required two-thirds majority to pass. In June 2005, Team San Jose built the South Hall, a $6.77 million, blue and white tent, adding 80,000 square feet (7,400 m2) of exhibit space",
          "qas": [
            {
              "question": "where is the businesses choosing to go?",
              "id": "1"
            },
            {
              "question": "how may votes did the ballot measure need?",
              "id": "2"
            },
            {
              "question": "By what year many Silicon Valley businesses were choosing the Moscone Center?",
              "id": "3
            }
          ]
        }
      ],
      "title": "Conference Center"
    }
  ]
}
```

### Preprocessing
Get parameters and convert input examples into features
```python
# preprocess input 
predict_file = 'inputs.json'

# Use read_squad_examples method from run_onnx_squad to read the input file
eval_examples = read_squad_examples(input_file=predict_file)

max_seq_length = 256
doc_stride = 128
max_query_length = 64
batch_size = 1
n_best_size = 20
max_answer_length = 30

vocab_file = os.path.join('uncased_L-12_H-768_A-12', 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

# Use convert_examples_to_features method from run_onnx_squad to get parameters from the input 
input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(eval_examples, tokenizer,
                                                                              max_seq_length, doc_stride, max_query_length)
```

### Output
The model produces the predictions/answers for the questions asked based on context from the given input paragraph. 

### Postprocessing
Write the predictions (answers to the questions) in a file. 
```python
# postprocess results
output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True)
output_prediction_file = os.path.join(output_dir, "predictions.json")
output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
write_predictions(eval_examples, extra_data, all_results,
                  n_best_size, max_answer_length,
                  True, output_prediction_file, output_nbest_file)
```

## Dataset
The model is trained with [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/) dataset that contains 100,000+ question-answer pairs on 500+ articles. 

## Validation accuracy
The accuracies obtained by the model on the validation set are mentioned above. The accuracies have been calculated on center cropped images with a maximum deviation of 1% (top-1 accuracy) from the paper.

## Training
Fine-tuned the model using SQuAD-1.1 dataset. Look at [BertTutorial.ipynb](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/BertTutorial.ipynb) for more information for converting the model from tensorflow to onnx and for fine-tuning

## Validation

## References
* **BERT** Model from the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

* [BERTtutorial](https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/BertTutorial.ipynb)
## Contributors
[Kundana Pillari](https://github.com/kundanapillari)

## License
Apache 2.0
