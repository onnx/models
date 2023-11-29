
# labels: name::transfoxllmhead author::transformers task::Generative_AI license::apache-2.0
from turnkeyml.parser import parse
from transformers import TransfoXLLMHeadModel, AutoConfig
import torch

# Parsing command-line arguments
pretrained, batch_size, max_seq_length = parse(
    ["pretrained", "batch_size", "max_seq_length"]
)

# Model and input configurations
if pretrained:
    model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")
else:
    config = AutoConfig.from_pretrained("transfo-xl-wt103")
    model = TransfoXLLMHeadModel(config)


inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
}


# Call model
model(**inputs)
    