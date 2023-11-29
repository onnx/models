
# labels: name::rag author::transformers task::Generative_AI license::apache-2.0
# Skip reason: Input error
from turnkeyml.parser import parse
from transformers import RagModel, AutoConfig
import torch

# Parsing command-line arguments
pretrained, batch_size, max_seq_length = parse(
    ["pretrained", "batch_size", "max_seq_length"]
)

# Model and input configurations
if pretrained:
    model = RagModel.from_pretrained("facebook/rag-token-base")
else:
    config = AutoConfig.from_pretrained("facebook/rag-token-base")
    model = RagModel(config)


inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
    