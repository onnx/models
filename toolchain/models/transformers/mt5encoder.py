
# labels: name::mt5encoder author::transformers task::Generative_AI license::apache-2.0
from turnkeyml.parser import parse
from transformers import MT5EncoderModel, AutoConfig
import torch

# Parsing command-line arguments
pretrained, batch_size, max_seq_length = parse(
    ["pretrained", "batch_size", "max_seq_length"]
)

# Model and input configurations
if pretrained:
    model = MT5EncoderModel.from_pretrained("google/mt5-small")
else:
    config = AutoConfig.from_pretrained("google/mt5-small")
    model = MT5EncoderModel(config)


inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
    