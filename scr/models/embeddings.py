import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from keras.layers import Embedding
from keras.initializers import Constant

def download_glove(version:str="6B"):
    valid_versions = ["6B", "42B.300d", "840B.300d", "twitter.27B"]
    if version not in valid_versions:
        raise ValueError(f"{version} is not a valid size({', '.join(valid_versions)})")
    
    filename= f"glove.{version}"
    file_zip = filename+".zip"
    file_txt = filename+".txt"
    url = f"https://downloads.cs.stanford.edu/nlp/data/{file_zip}"
    destination = "./models/embeddings/"
    
    if not os.path.isdir(destination):
        os.makedirs(destination)
    
    if not os.path.isfile(destination+file_txt):
        from utils import download_file, unzip_file
        download_file(url, destination+file_zip)
        unzip_file(destination+file_zip, destination+file_txt)
        os.remove(destination+file_zip)
        
def read_glove_embeddings(glove_file:str="./models/embeddings/glove.60B.100d.txt"):
    try:
        with open(glove_file, 'r', encoding="utf8") as file:       
            embedding_dict={}
            for line in tqdm(file, desc="Load Glove Embeddings"):
                values=line.split()
                word=values[0]
                vectors=np.asarray(values[1:],'float32')
                embedding_dict[word]=vectors
                
        return embedding_dict
        
    except (FileNotFoundError, pd.errors.ParserError) as e:
        print(f"Error reading glove file: {e}")
        return None

def create_embeddings_matrix(tokenizer, embedding_dict, embedding_dim):
    
    word_index = tokenizer.word_index
    print('Number of unique words:',len(word_index))

    num_words = len(word_index)+1
    embedding_matrix = np.zeros((num_words,  embedding_dim)) 

    for word, i in tqdm(word_index.items()):
        if i > num_words:
            continue
        
        emb_vec=embedding_dict.get(word)
        if emb_vec is not None:
            embedding_matrix[i]=emb_vec
            
    return embedding_matrix


def build_embeddings(embedding_matrix, max_text_length):
        
    if embedding_matrix is None:
        print("Error: Unable to load glove embeddings.")
        return None
    
    embedding_layer = Embedding(input_dim=embedding_matrix.shape[0], 
                               output_dim=embedding_matrix.shape[1], 
                               input_length=max_text_length,
                               embeddings_initializer=Constant(embedding_matrix),
                               trainable=False)
    
    return embedding_layer


if __name__ == "__main__":
    
    download_glove()