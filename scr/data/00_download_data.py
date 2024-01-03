'''
Download Disaster Tweets Kaggle Dataset
'''

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    # Create destination directories
    if not os.path.isdir('data/raw/'):
        os.makedirs('data/raw/')

    # Create a Kaggle API object
    api = KaggleApi()
    api.authenticate()
    # Download the competition data
    api.competition_download_files('nlp-getting-started')

    # Extract the downloaded zip file
    with zipfile.ZipFile('nlp-getting-started.zip', 'r') as zip_ref:
        zip_ref.extractall('data/raw/')

    # Remove the zip file
    os.remove('nlp-getting-started.zip')
    
if __name__=="__main__":
    main()
