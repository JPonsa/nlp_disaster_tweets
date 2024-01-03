import pandas as pd

from keras_nlp.models import BertTokenizer
from keras.preprocessing.text import Tokenizer

def build_tokenizer(x:str="standard"):
    if x == "standard":
        tokenizer = Tokenizer(oov_token="<OOV>")
    
    if x == "bert":
        tokenizer = BertTokenizer()
        
    return tokenizer

def preprocessing(text:pd.Series, tokenizer, len:int=32):
    from nltk.tokenize import word_tokenize
    from keras.utils import pad_sequences
    
    text = text.map(word_tokenize).to_list()
    sequences = tokenizer.texts_to_sequences(text)
    text_pad = pad_sequences(sequences, maxlen=len, truncating='post', padding='post')
    return text_pad