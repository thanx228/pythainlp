# -*- coding: utf-8 -*-
from typing import List

from ssg import syllable_tokenize


def segment(text: str) -> List[str]:
    """
    Syllable tokenizer using ssg
    """
    return [] if not text or not isinstance(text, str) else syllable_tokenize(text)
