'''
DISCLAIMER 
This file was originally created by the developers of the DeepVoice3 repository and is used to preprocess input data.
'''

# coding: utf-8
from text.symbols import symbols

import nltk
from random import random

n_vocab = len(symbols)

_arpabet = nltk.corpus.cmudict.dict()

# Assigning pronounciation to the text 
def _maybe_get_arpabet(word, p):
    try:
        phonemes = _arpabet[word][0]
        phonemes = " ".join(phonemes)
    except KeyError:
        return word

    return '{%s}' % phonemes if random() < p else word

# Splits the words in the input text 
def mix_pronunciation(text, p):
    text = ' '.join(_maybe_get_arpabet(word, p) for word in text.split(' '))
    return text

# Decodes the input text into valid input features for unique pronunciation
def text_to_sequence(text, p=0.0):
    if p >= 0:
        text = mix_pronunciation(text, p)
    from text import text_to_sequence
    text = text_to_sequence(text, ["english_cleaners"])
    return text



