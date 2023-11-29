
# labels: name::esm author::transformers task::Generative_AI license::apache-2.0
from turnkeyml.parser import parse
from transformers import EsmModel, AutoConfig
import torch

# Parsing command-line arguments
pretrained, batch_size, max_seq_length = parse(
    ["pretrained", "batch_size", "max_seq_length"]
)

# Model and input configurations
if pretrained:
    model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
else:
    config = AutoConfig.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = EsmModel(config)

# Make sure the user's sequence length fits within the model's maximum
assert max_seq_length <= model.config.max_position_embeddings


inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
    