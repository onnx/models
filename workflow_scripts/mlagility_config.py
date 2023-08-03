models_info = [
    # PyTorch models
    "torch_hub/alexnet.py",
    "torch_hub/resnet50.py",
    "torchvision/maskrcnn_resnet50_fpn.py",
    "torchvision/ssd300_vgg16.py",
    "torch_hub/mobilenet_v2.py",
    "torch_hub/mobilenet_v3_large.py",
    "torch_hub/resnet18.py",
    "torch_hub/resnet34.py",
    "torch_hub/resnet101.py",
    "torch_hub/vgg16.py",
    "torch_hub/vgg16_bn.py",
    "torch_hub/vgg19.py",
    "torch_hub/vgg19_bn.py",
    "torch_hub/densenet121.py",
    "torch_hub/inception_v3.py",
    "torch_hub/googlenet.py",
    # Transformer models
    "popular_on_huggingface/openai_clip-vit-large-patch14.py",
    "popular_on_huggingface/distilbert-base-uncased.py",
    "popular_on_huggingface/distilbert-base-multilingual-cased.py",

    "popular_on_huggingface/CompVis_stable-diffusion-v1-4.py",
    "popular_on_huggingface/DMetaSoul_sbert-chinese-general-v2.py",
    "popular_on_huggingface/deepset_deberta-v3-base-squad2.py",
    "popular_on_huggingface/Musixmatch_umberto-commoncrawl-cased-v1.py",
    "popular_on_huggingface/aiknowyou_aiky-sentence-bertino.py",
    "popular_on_huggingface/deepset_deberta-v3-base-squad2.py",
    "popular_on_huggingface/facebook_deit-base-patch16-224.py",
    "popular_on_huggingface/microsoft_beit-large-patch16-384.py",
    # Failed models
    """
    "transformers/bert_generation.py", # non consistent created model from mlagility
    "popular_on_huggingface/xlm-roberta-base.py", # output nan
    "popular_on_huggingface/roberta-base.py", # output nan
    "popular_on_huggingface/distilroberta-base.py", # output nan
    "popular_on_huggingface/albert-base-v2", # Status Message: indices element out of data bounds, idx=8 must be within the inclusive range [-2,1]
    "transformers/bert_generation.py", # non consistent created model from mlagility
    "popular_on_huggingface/bert-base-uncased.py", # failed to create test data
    "popular_on_huggingface/xlm-roberta-large.py", # failed to create test data
    "popular_on_huggingface/bert-large-uncased.py", # failed to create test data
    "popular_on_huggingface/AI-Growth-Lab_PatentSBERTa.py", # failed to create test data
    "popular_on_huggingface/BM-K_KoSimCSE-roberta.py", # failed to create test data
    "popular_on_huggingface/Babelscape_wikineural-multilingual-ner.py", # failed to create test data
    "popular_on_huggingface/Bhuvana_t5-base-spellchecker.py", # failed to create test data
    "popular_on_huggingface/CAMeL-Lab_bert-base-arabic-camelbert-ca-pos-egy.py", # failed to create test data
    "popular_on_huggingface/CompVis_ldm-text2im-large-256.py", # failed to create test data
    "popular_on_huggingface/Davlan_bert-base-multilingual-cased-masakhaner.py", # failed to create test data
    "popular_on_huggingface/DmitryPogrebnoy_MedRuRobertaLarge.py", # output nan
    "popular_on_huggingface/Elron_bleurt-base-512.py", # failed to create test data
    "popular_on_huggingface/FinanceInc_finbert_fls.py", # failed to create test data
    "popular_on_huggingface/HooshvareLab_bert-fa-zwnj-base-ner.py", # failed to create test data
    "popular_on_huggingface/AmazonScience_qanlu.py", # output nan
    "popular_on_huggingface/ElKulako_cryptobert.py", # output nan
    "popular_on_huggingface/cardiffnlp_twitter-xlm-roberta-base-sentiment.py", # output nan
    "popular_on_huggingface/roberta-base-squad2-distilled.py", # output nan
    "popular_on_huggingface/efederici_cross-encoder-umberto-stsb.py", # output nan
    "popular_on_huggingface/nickprock_xlm-roberta-base-banking77-classification.py", # output nan
    "popular_on_huggingface/sentence-transformers_nli-roberta-large.py", # output nan
    "popular_on_huggingface/sentence-transformers_paraphrase-mpnet-base-v2.py", # output nan
    """
]
