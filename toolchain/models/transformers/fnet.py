
# labels: name::fnet author::transformers task::Generative_AI license::apache-2.0
from turnkeyml.parser import parse
from transformers import FNetModel, AutoConfig
import torch

# Parsing command-line arguments
pretrained, batch_size, max_seq_length = parse(
    ["pretrained", "batch_size", "max_seq_length"]
)

# Model and input configurations
if pretrained:
    model = FNetModel.from_pretrained("google/fnet-base")
else:
    config = AutoConfig.from_pretrained("google/fnet-base")
    model = FNetModel(config)

# Make sure the user's sequence length fits within the model's maximum
assert max_seq_length <= model.config.max_position_embeddings


inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
}


# Call model
model(**inputs)
    