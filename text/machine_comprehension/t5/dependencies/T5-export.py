from transformers import T5ForConditionalGeneration
from .models import CombinedDecoder, SimplifiedT5Encoder
import torch


def create_t5_encoder_decoder(pretrained_version='t5-base'):
    """ Generates an encoder and a decoder model with a language model head from a pretrained huggingface model

    Args:
        pretrained_version (str): Name of a pretrained model, or path to a pretrained / finetuned version of T5

    Returns:
        simplified_encoder: pytorch t5 encoder with a wrapper to output only the hidden states
        decoder_with_lm_head: pytorch t5 decoder with a language modeling head
    """

    # T5 is an encoder / decoder model with a language modeling head on top.
    # We need to separate those out for efficient language generation
    model = T5ForConditionalGeneration.from_pretrained(pretrained_version)

    return turn_model_into_encoder_decoder(model)

def turn_model_into_encoder_decoder(model):
    encoder = model.encoder
    decoder = model.decoder
    lm_head = model.lm_head

    decoder_with_lm_head = CombinedDecoder(decoder, lm_head, model.config)
    simplified_encoder = SimplifiedT5Encoder(encoder)

    return simplified_encoder, decoder_with_lm_head


def generate_onnx_representation(pretrained_version=None, output_prefix=None, model=None):
    """ Exports a given huggingface pretrained model, or a given model and tokenizer, to onnx

    Args:
        pretrained_version (str): Name of a pretrained model, or path to a pretrained / finetuned version of T5
        output_prefix (str): Path to the onnx file
    """
    if (pretrained_version is None or output_prefix is None) and model is None:
        print("You need to specify both pretrained_version (the pretrained model you wish to export) and output_prefix"
              "(the path you want to export to). Alternatively you can export a model you have in memory.")
        return
    if model is not None:
        # Transform model into encoder and decoder with lm head
        simplified_encoder, decoder_with_lm_head = turn_model_into_encoder_decoder(model)
    else:
        # Loading model_data
        simplified_encoder, decoder_with_lm_head = create_t5_encoder_decoder(pretrained_version)

    # Example sequence
    input_ids = torch.tensor([[42] * 10])

    # Exports to ONNX
    _ = torch.onnx.export(
                            decoder_with_lm_head,
                            (input_ids, simplified_encoder(input_ids)),
                                   f"{output_prefix}-decoder-with-lm-head.onnx",
                                   export_params=True,
                            opset_version=12,
                            input_names=['input_ids', 'encoder_hidden_states'],
                            output_names=['hidden_states'],
                            dynamic_axes={
                              'input_ids': {0:'batch', 1: 'sequence'},
                              'encoder_hidden_states': {0:'batch', 1: 'sequence'},
                              'hidden_states': {0:'batch', 1: 'sequence'},
                            })

    _ = torch.onnx._export(
                            simplified_encoder,
                                   input_ids,
                                   f"{output_prefix}-encoder.onnx",
                                   export_params=True,
                            opset_version=12,
                            input_names=['input_ids'],
                            output_names=['hidden_states'],
                            dynamic_axes={
                               'input_ids': {0:'batch', 1: 'sequence'},
                               'encoder_hidden_states': {0:'batch', 1: 'sequence'},
                               'hidden_states': {0:'batch', 1: 'sequence'},
                            }
                           )
