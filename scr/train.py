import pandas as pd



simpleNN = ["LSTM", "GRU", "RNN"]
llms = ["DistilBert",]
implemented_models = simpleNN + llms

def create_emmbedings(tokenizer, max_text_length):

    from models.embeddings import read_glove_embeddings
    from models.embeddings import create_embeddings_matrix
    from models.embeddings import build_embeddings
    
    embeddings_dim = 100
    glove_embeddings_dict = read_glove_embeddings("./models/embeddings/glove.6B/glove.6B.100d.txt")
    embeddings_matrix = create_embeddings_matrix(tokenizer, glove_embeddings_dict, embeddings_dim)
    embeddings_layer = build_embeddings(embeddings_matrix, max_text_length)
    
    return embeddings_layer

if __name__ == "__main__":
    
    from argparse import ArgumentParser
    from datetime import datetime
    import pickle
    
    import pandas as pd
    import numpy as np
    import optuna
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    import mlflow
    from mlflow.models import infer_signature
    import mlflow.keras
    
    from models.models import train_model
    from models.hyperparams import MAX_LEN, MLFLOW_EXPERIMENT
   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser =  ArgumentParser()
    parser.add_argument("--model", type=str, default="LSTM", choices=implemented_models)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tune-hyperparams", action="store_true")
    parser.add_argument("--hyperparam-trials", type=int, default=10)
    parser.add_argument("--experiment", type=str, default=None)
    
    args = parser.parse_args()
    

    mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

    run_name= f"{args.model}_{timestamp}"
    with mlflow.start_run(run_name=run_name):
    
        clean = pd.read_csv("./data/clean/train.csv")
        clean = clean.dropna()

        X= clean["text"]
        y = clean["target"].values

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, random_state=42)

        if args.model in simpleNN:
            
            from scr.models.preprocessing import build_tokenizer, preprocessing
            tokenizer = build_tokenizer("standard")
            tokenizer.fit_on_texts(X)
            
            with open('./models/tokenizer.pkl', 'wb') as file:
                pickle.dump(tokenizer, file)
            
            mlflow.log_artifact("tokenizer.pkl", artifact_path="preprocessing")
            
            embeddings_layer = create_emmbedings(tokenizer, MAX_LEN)
            
            X_train = preprocessing(X_train, tokenizer, MAX_LEN)
            X_val = preprocessing(X_val, tokenizer, MAX_LEN)
            
            from models.models import build_NN
            
            # Optimize NN
            if args.tune_hyperparams:
                
                from scr.models.hyperparams import objective
                
                study = optuna.create_study(
                    direction="maximize",
                    storage=f"sqlite:///{MLFLOW_EXPERIMENT}_simpleNN.optuna.db.sqlite3",
                    study_name=MLFLOW_EXPERIMENT,
                    load_if_exists=True,
                    )

                # Optimize the objective function
                study.optimize(objective, n_trials=args.hyperparam_trials, show_progress_bar=True)

                model = build_NN(study.best_params)
            
            else:
                model = build_NN(embeddings_layer, 2, args.model)
        
        elif args.model in llms:
            from models.models import build_DistilBert
            model = build_DistilBert()
        
       
        mlflow.log_params(vars(args))

        trained_model, history = train_model(model, 
                                             X_train, 
                                             y_train, 
                                             X_val, 
                                             y_val, 
                                             epoch=args.epoch, 
                                             batch_size=args.batch_size)
        
        y_pred = trained_model.predict(X_val)
        signature = infer_signature(X_val, y_pred)
        
        y_pred = np.argmax(y_pred, axis=1)
        
        metrics = {"acc":accuracy_score(y_val, y_pred),
                   "precision":precision_score(y_val, y_pred),
                   "recall":recall_score(y_val, y_pred),
                   "f1":f1_score(y_val, y_pred)}
        
        mlflow.log_metrics(metrics)
        mlflow.keras.log_model(trained_model,"models", signature=signature, registered_model_name= args.model)