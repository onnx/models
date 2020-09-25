# Silero Speech To Text

## Description

Silero Speech-To-Text models provide enterprise grade STT in a compact form-factor for several commonly spoken languages. Unlike conventional ASR models our models are robust to a variety of dialects, codecs, domains, noises, lower sampling rates (for simplicity audio should be resampled to 16 kHz). The models consume a normalized audio in the form of samples (i.e. without any pre-processing except for normalization to -1 … 1) and output frames with token probabilities. We provide a decoder utility for simplicity (we could include it into our model itself, but it is hard to do with ONNX for example).

We hope that our efforts with Open-STT and Silero Models will bring the ImageNet moment in speech closer.

## Use Cases

Transcribing speech into text. Please see detailed benchmarks for various domains [here](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks).

## Model

Please note that models are downloaded automatically with the utils provided below.
| Model           | Download                                                                                       | ONNX version | Opset version |
|-----------------|:-----------------------------------------------------------------------------------------------|:-------------|:--------------|
| English (en_v1) | [174 MB](https://silero-models.ams3.cdn.digitaloceanspaces.com/models/en/en_v1_batchless.onnx) | 1.7.0        | 12            |
| German  (de_v1) | [174 MB](https://silero-models.ams3.cdn.digitaloceanspaces.com/models/de/de_v1_batchless.onnx) | 1.7.0        | 12            |
| Spanish (es_v1) | [201 MB](https://silero-models.ams3.cdn.digitaloceanspaces.com/models/es/es_v1_batchless.onnx) | 1.7.0        | 12            |
| Model list      | [0 MB](https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml)             | 1.7.0        | 12            |

### Source

Original implementation in PyTorch => simplification => TorchScript => ONNX.

## Inference

We try to simplify starter scripts as much as possible using handy torch.hub utilities.

```bash
pip install -q torch torchaudio omegaconf soundfile onnx onnxruntime
```

```python
import onnx
import torch
import onnxruntime
from omegaconf import OmegaConf

language = 'en' # also available 'de', 'es'

# load provided utils
_, decoder, utils = torch.hub.load(github='snakers4/silero-models', model='silero_stt', language=language)
(read_batch, split_into_batches,
 read_audio, prepare_model_input) = utils

# see available models
torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml', 'models.yml')
models = OmegaConf.load('models.yml')
available_languages = list(models.stt_models.keys())
assert language in available_languages

# load the actual ONNX model
torch.hub.download_url_to_file(models.stt_models.en.latest.onnx, 'model.onnx', progress=True)
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession('model.onnx')

# download a single file, any format compatible with TorchAudio (soundfile backend)
torch.hub.download_url_to_file('https://opus-codec.org/static/examples/samples/speech_orig.wav', dst ='speech_orig.wav', progress=True)
test_files = ['speech_orig.wav']
batches = split_into_batches(test_files, batch_size=10)
input = prepare_model_input(read_batch(batches[0]))

# actual onnx inference and decoding
onnx_input = input.detach().cpu().numpy()[0]
ort_inputs = {'input': onnx_input}
ort_outs = ort_session.run(None, ort_inputs)
decoded = decoder(torch.Tensor(ort_outs[0]))
print(decoded)
```

## Dataset (Train)

Not disclosed by model authors.

## Validation

We have performed a vast variety of benchmarks on different publicly available validation datasets. Please see benchmarks [here]([here](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks)). We neither own these datasets nor we provide mirrors for them or re-upload them for legal reasons.

It is [customary](https://github.com/syhw/wer_are_we) for English STT models to report metrics on Librispeech. Please beware though that these metrics have very little in common with real life / production metrics and with model generalization (see [here](https://blog.timbunce.org/2019/02/11/a-comparison-of-automatic-speech-recognition-asr-systems-part-2/), and [here](https://thegradient.pub/a-speech-to-text-practitioners-criticisms-of-industry-and-academia/#criticisms-of-academia) section "Sample Inefficient Overparameterized Networks Trained on "Small" Academic Datasets"). Hence we report metrics compared to a premium Google STT API (heavily abridged).

### EN V1

| Dataset                              | Silero CE | Google Video Premium | Google Phone Premium |
|--------------------------------------|-----------|----------------------|----------------------|
| **AudioBooks**                       |           |                      |                      |
| en_v001_librispeech_test_clean       | 8.6       | 7.8                  | 8.7                  |
| en_librispeech_val                   | 14.4      | 11.3                 | 13.1                 |
| en_librispeech_test_other            | 20.6      | 16.2                 | 19.1                 |

Please see benchmarks [here](https://github.com/snakers4/silero-models/wiki/Quality-Benchmarks) for more details.

## References

- [Silero Models](https://github.com/snakers4/silero-models)
- [Alexander Veysov, "Toward's an ImageNet Moment for Speech-to-Text", The Gradient, 2020](https://thegradient.pub/towards-an-imagenet-moment-for-speech-to-text/)
- [Alexander Veysov, "A Speech-To-Text Practitioner’s Criticisms of Industry and Academia", The Gradient, 2020](https://thegradient.pub/a-speech-to-text-practitioners-criticisms-of-industry-and-academia/)

## Contributors

[Alexander Veysov](http://github.com/snakers4) together with Silero AI Team.

## License

AGPL-3.0 License
