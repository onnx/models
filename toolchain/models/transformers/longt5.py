
# labels: name::longt5 author::transformers task::Generative_AI license::apache-2.0
from turnkeyml.parser import parse
from transformers import LongT5Model, AutoConfig
import torch

# Parsing command-line arguments
pretrained, batch_size, max_seq_length = parse(
    ["pretrained", "batch_size", "max_seq_length"]
)

# Model and input configurations
if pretrained:
    model = LongT5Model.from_pretrained("google/long-t5-local-base")
else:
    config = AutoConfig.from_pretrained("google/long-t5-local-base")
    model = LongT5Model(config)


inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "decoder_input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
    