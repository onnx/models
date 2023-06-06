import torch
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("textattack/distilbert-base-cased-CoLA")
torch.onnx.export(model.base_model, model.dummy_inputs, "distilbert-18.onnx", verbose=True, input_names=["input_ids", "attention_mask"], output_names=["logits"], opset_version=11, dynamic_axes={"input_ids": [0, 1], "attention_mask": [0, 1], "logits": [0, 1]})