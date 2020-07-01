# DeepVoice3

## Description
Deep learning model that performs end-to-end neural speech synthesis. Requires fewer parameters and is faster than other state-of-the-art neural speech synthesis systems.

## Model

|Model        |Download  | Download (with sample test data)|ONNX version|Opset version|Accuracy|
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|DeepVoice3      | [52.8 MB](audio/deepvoice3/model/deepvoice3-11.onnx) | [49.1 MB](audio/deepvoice3/model/deepvoice3-11.tar.gz) |1.6| 11|Speech naturalness of 95.5 Pronunciation accuracy of 95.1 |

### Source
Pytorch => onnx model

## Inference
Used [ONNX Runtime](https://github.com/microsoft/onnxruntime) to perform inference

### Input
Sequence of words as a string in a list. Example: texts = ["Thank you very much.", "Hello.", "Deep voice 3."]. Inputs texts are converted into features including sequence resized to the shape [3,21] of type int64 and text_positions to the shape [3,12,80] of type float32. 

### Preprocessing
A few actions were taken on the text to improve the quality of the audio prediction:
* All Uppercase letters.
* Punctuation marks removed.
* Every utterance ended with a period or a question mark.
* Spaces replaced between words with separator characters that represent the time between words uttered by the speaker.

Get parameters and convert textual features to an internal representation.
``` 
sequence = np.array(frontend.text_to_sequence(text, p=p))
sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)
speaker_ids = None if speaker_id is None else torch.LongTensor([speaker_id]).to(device)
```

### Output
Generates audio output. The decoder uses mel-band log-magnitude spectograms as audio frame representation.

### Postprocessing
Print text with generated audio. 
```
for idx, text in enumerate(texts):
  print(idx, text)
  tts(model, text, figures=False)
```

## Model Creation

### Dataset (Train and validation)
The original model is pretrained with the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset. This is a public domain speech dataset consisting of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books.

### Training
The pretrained model is referenced in the [deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch) repository. 

### Validation accuracy
Speech naturalness of 95.5 and Pronunciation accuracy of 95.1 computed over LJSpeech dataset.
<hr>

### References
Deep Voice3 model from the paper [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654)

## Contributors
Kundana Pillari 

## License
MIT "Expat" License 
<hr>
