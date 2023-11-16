
# labels: name::jukebox author::transformers task::Generative_AI license::apache-2.0
# Skip reason: Input Error
from turnkeyml.parser import parse
from transformers import JukeboxModel, AutoConfig
import torch

# Parsing command-line arguments
pretrained, batch_size, max_seq_length = parse(
    ["pretrained", "batch_size", "max_seq_length"]
)

# Model and input configurations
if pretrained:
    model = JukeboxModel.from_pretrained("openai/jukebox-1b-lyrics")
else:
    config = AutoConfig.from_pretrained("openai/jukebox-1b-lyrics")
    model = JukeboxModel(config)


inputs = {
    "input_ids": torch.ones(batch_size, max_seq_length, dtype=torch.long),
    "attention_mask": torch.ones(batch_size, max_seq_length, dtype=torch.float),
}


# Call model
model(**inputs)
    