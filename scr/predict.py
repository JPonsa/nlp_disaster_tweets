simpleNN = ["LSTM", "GRU", "RNN"]
llms = ["DistilBert",]
implemented_models = simpleNN + llms

if __name__ == "__main__":

    from argparse import ArgumentParser
    import os
    import pickle
    
    import numpy as np
    import pandas as pd
    
    from mlflow.tracking import MlflowClient
    import mlflow.pyfunc
    
    from models.preprocessing import preprocessing
    from models.hyperparams import MAX_LEN
    
    parser =  ArgumentParser()
    parser.add_argument("--model", type=str, default="LSTM", choices=implemented_models)
    parser.add_argument("--features_dir", type=str, default="./data/clean/test.csv")
    parser.add_argument("--outcome_dir", type=str, default="./outcome/y_pred.csv")
    
    args = parser.parse_args()

    client = MlflowClient()

    model_name = args.model
    stage = "Production"

    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == stage:
            model_version = mv.version   
     
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    
 
    
    preprocessing_path = "./models/tokenizer.pkl"
    with open(preprocessing_path, "rb") as file:
        tokenizer = pickle.load(file)
    
    df = pd.read_csv(args.features_dir)
    X = df["text"].dropna()

    X = preprocessing(X, tokenizer, MAX_LEN)
    
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)

    np.savetxt(args.outcome_dir, y_pred, delimiter=",")
