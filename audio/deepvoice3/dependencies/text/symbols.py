'''
DISCLAIMER 
This file was originally created by the developers of the DeepVoice3 repository and is used to preprocess input data.
'''

from .cmudict import valid_symbols

_pad = '_'
_eos = '~'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) + _arpabet
