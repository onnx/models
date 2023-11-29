
# labels: name::bloom author::transformers task::Generative_AI license::apache-2.0
from turnkeyml.parser import parse
from transformers import BloomModel, AutoConfig
import torch

# Parsing command-line arguments
pretrained, batch_size, max_seq_length = parse(
    ["pretrained", "batch_size", "max_seq_length"]
)

# Model and input configurations
if pretrained:
    model = BloomModel.from_pretrained("bigscience/bloom-560m")
else:
    config = AutoConfig.from_pretrained("bigscience/bloom-560m")
    model = BloomModel(config)


inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
    