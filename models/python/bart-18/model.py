# labels: test_group::mlagility name::bert author::transformers task::Natural_Language_Processing
"""
https://huggingface.co/docs/transformers/v4.26.1/en/model_doc/bert#overview
"""
from mlagility.parser import parse
import transformers
import torch

torch.manual_seed(0)

# Parsing command-line arguments
batch_size, max_seq_length = parse(["batch_size", "max_seq_length"])


# Model and input configurations
config = transformers.BertConfig()
model = transformers.BertModel(config)
inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)

torch.onnx.export(model.base_model, inputs, "bart-18.onnx", opset_version=18)
