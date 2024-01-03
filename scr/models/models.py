'''
Models from https://github.com/Isoken00/-Fake-News-Classification-in-Python/blob/main/Fakenews.ipynb


'''
import pandas as pd

from keras.layers import LSTM, Dense, Dropout, GRU, SimpleRNN, Lambda
from keras.models import Sequential
from keras.losses import SparseCategoricalCrossentropy
from keras_nlp.models import DistilBertPreprocessor, DistilBertClassifier
import tensorflow as tf


def compile_model(func):
    def wrapper_func(*args, **kwargs):
        # Do something before the function.
        model = func(*args, **kwargs)
        # Do something after the function.
        model.compile(loss= SparseCategoricalCrossentropy(from_logits=False),
                      optimizer = tf.keras.optimizers.Adam(1e-5),
                      metrics=['accuracy'])
        
        return model
    return wrapper_func


# TODO: Add preprocessing within the NN model

@compile_model
def build_NN(embedding_layer, 
             num_classes:int=2, 
             flavour:str="LSTM", 
             n_hidden_layers:int=1,
             neurons_1st_layer:int=100,
             neurons_hidden_layer:int=32,
             dropout_rate:float=0.3
             ):
    
    implemented = ["LSTM", "GRU", "RNN"]
    if flavour not in implemented:
        raise NotImplementedError(f"{flavour} still not implemented. Try {implemented}")

    model = Sequential()
    model.add(embedding_layer)
    
    if flavour == "LSTM":
        model.add(LSTM(neurons_1st_layer))
        
    if flavour == "GRU":
        model.add(GRU(neurons_1st_layer))
    
    if flavour == "RNN":
        model.add(SimpleRNN(neurons_1st_layer))
    
    for i in range(n_hidden_layers):
        model.add(Dropout(dropout_rate))
        model.add(Dense(neurons_hidden_layer, activation='relu'))
   
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

@compile_model
def build_DistilBert(num_classes:int=2, seq_length:int=64):
    """
    Model as described in https://www.kaggle.com/code/alexia/kerasnlp-starter-notebook-disaster-tweets

    Returns
    -------
        model
    """
    
    # Load a DistilBERT model.
    preset= "distil_bert_base_en_uncased"

    # Use a shorter sequence length.
    preprocessor = DistilBertPreprocessor.from_preset(preset,sequence_length=seq_length, name="preprocessor_4_tweets")

    # Pretrained classifier.
    model = DistilBertClassifier.from_preset(preset,preprocessor = preprocessor, num_classes=num_classes, activation="softmax")
    
    return model


def train_model(model, X_train, y_train, X_val=None, y_val=None,epoch:int=20, batch_size:int=256):
    
    # Train model with Train and test set data
    # Number of epochs, batch size as minimum parameter
    if X_val is None:
        history = model.fit(X_train, y_train, epochs=epoch, batch_size = batch_size ,validation_split=0.2)#validation_data=(X_test, y_test)) 
    else:
        history = model.fit(X_train, y_train, epochs=epoch, batch_size = batch_size , validation_data=(X_val, y_val)) 

    return model, history

