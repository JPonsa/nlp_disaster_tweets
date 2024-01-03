
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import date
import os
import matplotlib.pyplot as plt

def performance_history(history, name, directory_path):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    # Save the plot
    save_path = os.path.join(directory_path, f'{name}.performance.jpeg')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    
    # Display the plot
    plt.show() 

def model_evaluation(model,X_test,y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
    return score

def store_model(model, model_type, name, directory_path=None):
    # Existing code

    # Add the following condition to check if directory_path is provided
    if directory_path:
        save_path = os.path.join(directory_path, f'Model_{model_type}_{name}.h5')
    else:
        save_path = f'Model_{model_type}_{name}.h5'
    
    model.save(save_path)
    print(f'Model \'{model_type}_{name}\' saved at: {save_path}')

def performance_report(model, X_test, y_test, directory_path, name):

    time = date.today()
 
    yhat_probs = model.predict(X_test, verbose=0)
    yhat_classes = (yhat_probs > 0.5).astype('int32')

    # Now yhat_classes contains your binary predictions

    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    print(f'Accuracy: {accuracy}')
    
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print(f'Precision: {precision}')
    
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print(f'Recall: {recall}')
    
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print(f'F1 score: {f1}')

    if os.path.isfile(os.path.join(directory_path, 'report.csv')):
        report_df = pd.read_csv(os.path.join(directory_path, 'report.csv'), index_col=0)
    else:
        report_df = pd.DataFrame(
                columns=['time', 'name', 'Precision', 'Recall', 'f1_score', 'accuracy'])

    new_row = pd.DataFrame({'time': [time], 'name': [name], 'Precision': [precision], 'Recall': [recall], 'f1_score': [f1], 'accuracy': [accuracy]})

    # Exclude empty or all-NA columns before concatenation
    report_df = report_df.dropna(axis=1, how='all')
    new_row = new_row.dropna(axis=1, how='all')

    report_df = pd.concat([report_df, new_row], ignore_index=True)
  
    report_df.to_csv(os.path.join(directory_path, 'report.csv'), index=False)