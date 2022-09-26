# -*- coding: utf-8 -*-
"""
esupar: Tokenizer POS-tagger and Dependency-parser with BERT/RoBERTa/DeBERTa models for Japanese and other languages

GitHub: https://github.com/KoichiYasuoka/esupar
"""
from typing import List, Union
try:
    import esupar
except ImportError:
    raise ImportError("Import Error; Install esupar by pip install esupar")


class Parse:
    def __init__(self, model: str="th") -> None:
        if model is None:
            model = "th"
        self.nlp=esupar.load(model)

    def __call__(self, text: str, tag: str="str") -> Union[List[List[str]], str]:
        _data = str(self.nlp(text))
        return [i.split() for i in _data.splitlines()] if tag =="list" else _data
