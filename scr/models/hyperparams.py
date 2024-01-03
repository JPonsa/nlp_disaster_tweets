import optuna
from sklearn.metrics import f1_score

from models.models import train_model

MAX_LEN = 32
MLFLOW_EXPERIMENT = "nlp_disaster_tweets"


def sample_hyperams(trial:optuna.trial.Trial, 
                    model_flavour:str="LSTM"):
    
    params = {"flavour" : model_flavour,
              "n_hidden_layers": trial.suggest_int("n_hidden_layers", 1, 5),
              "neurons_1st_layer": trial.suggest_int("neurons_1st_layer", 50, 150, steps=25),
              "neurons_hidden_layer": trial.suggest_int("neurons_hidden_layer", 32, int(32*4), steps=32),
              "dropout_rate":trial.suggest_float("dropout_rate", 0.0, 0.3, steps=0.1)
    }
    
    return params


def objective(trial, model_flavour):
    params = sample_hyperams(model_flavour)
    
    trained, history = train_model(build_NN(**params), X_test, y_test)
    y_pred = trained.predict(X_val)
    
    return f1_score(y_val, y_pred)