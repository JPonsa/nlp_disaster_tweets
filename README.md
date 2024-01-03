# Disaster Tweets

## Summary

This project aims to leverage advanced natural language processing (NLP) techniques using DistilBert, GRU (Gated Recurrent Unit), LSTM (Long Short-Term Memory), and RNN (Recurrent Neural Network) to analyze the Disaster Tweets dataset. The primary goal is to develop a model that can accurately classify tweets as either related to a disaster or not. The project utilizes state-of-the-art (SOTA) deep learning models and explores their performance in handling text classification tasks.

## Goals
- Implement and compare the performance of DistilBert, GRU, LSTM, and RNN for disaster tweet classification.<img src="https://upload.wikimedia.org/wikipedia/commons/8/8c/White_check_mark_in_dark_green_rounded_square.svg" widht="15" height="15"/>
- Develop a robust and accurate model for identifying tweets related to disasters.<img src="https://upload.wikimedia.org/wikipedia/commons/8/8c/White_check_mark_in_dark_green_rounded_square.svg" widht="15" height="15"/>
- Explore the strengths and weaknesses of different architectures in the context of natural language processing.<img src="https://upload.wikimedia.org/wikipedia/commons/8/8c/White_check_mark_in_dark_green_rounded_square.svg" widht="15" height="15"/>

## About the data
The [Kaggle dataset](https://www.kaggle.com/competitions/nlp-getting-started/data) used in this project consists of tweets labeled as either disaster-related or non-disaster-related. Each tweet is associated with a binary label indicating whether it is relevant to a disaster or not. The data exploration process will involve understanding the distribution of classes, preprocessing text data, and preparing it for model training.

## Approach

### Data Preprocessing:
- Tokenization, padding, and leveraging GloVe embeddings for word representation.
- Removal of frequent words.

### Model Architecture:
- DistilBert: Utilizing a pre-trained transformer model for contextualized embeddings.
 - GRU, LSTM, and RNN: Employing recurrent neural network architectures for sequence modeling.

### Model Evaluation
- Model evaluation using metrics such as accuracy, precision, recall, and F1 score
![Training Curve](https://github.com/JPonsa/nlp_disaster_tweets/blob/main/figures/RNN.performance.jpeg)
![Confusion Matrix](https://github.com/JPonsa/nlp_disaster_tweets/blob/main/figures/distilBert.confusion_matrix.png)

## Observations

- Simple Neural Networks (e.g. Single LSTM, GRU or Recurrent layer) have a similar performance than more complex models like DistilBert
![model comparison](https://github.com/JPonsa/nlp_disaster_tweets/blob/main/figures/Model.competition.png)

## Future Works:
- Implement it using cloud computing. This will allow me to test more complex models and possibly achieve better performance. E.g increase embedding length, more complex NN architectures.
- Try other embeddings. E.g. fasttext
- Use Optuna to perform hyperparameter tunning in the NN, including how many hidden layers and neurons.
- Embeddings preprocessing within the model artifact. Similar to sklearn pipelines.

# References:
Fake news classification: Definition Several models
https://github.com/Isoken00/-Fake-News-Classification-in-Python/tree/main
