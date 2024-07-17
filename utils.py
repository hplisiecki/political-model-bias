import pandas as pd
import os



def check_max_token_length(tokenizer, texts):
    if isinstance(texts, str):
        texts = [texts]
    max_len = 0
    for text in texts:
        token_length = len(tokenizer.encode(text))
        if  token_length > max_len:
            max_len = token_length
    return max_len