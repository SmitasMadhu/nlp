from datetime import datetime
from typing import List
import sys
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import pandas as pd
import numpy as np
import re
import json
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from bs4 import BeautifulSoup
#from datetime import datetime
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sentimental_model import __version__ as _version
from sentimental_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

def strip_html(text):
    # BeautifulSoup is a useful library for extracting data from HTML and XML documents
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_punctuations(text):

    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern,'',text)

    # Single character removal
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

    # Removing multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text

def populate_stop_words():
    stopword_list = nltk.corpus.stopwords.words('english')
    updated_stopword_list = []

    for word in stopword_list:
        if word=='not' or word.endswith("n't"):
            pass
        else:
            updated_stopword_list.append(word)
            
    addl_remove_list = ['mustn', 'doesn', "shouldn", "couldn", "didn", "aren", "haven", "mightn", "needn", "shan", "hasn", "isn", "weren"]

    for item in addl_remove_list:
        updated_stopword_list.remove(item)
    
    print("number of stopwords ", len(updated_stopword_list))
    return updated_stopword_list


def remove_stopwords_and_lemmatize(text):
    #instantiate Lemmatizer
    lmtizer = WordNetLemmatizer()
     # splitting strings into tokens (list of words)
    tokens = nltk.tokenize.word_tokenize(text)  
    tokens = [token.strip() for token in tokens]
    #remove stopwords
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    #lemmatize the tokens and create the sentence back
    filtered_text = ' '.join([lmtizer.lemmatize(word, pos ='a') for word in filtered_tokens])
    
    return filtered_text

def create_and_update_sentiment_field(input_df) :

    input_df['Sentiment'] = input_df['Score'].apply(lambda x: 'Positive' if x >= 3 else 'Negetive')
    print("value counts")
    print(input_df['Sentiment'].value_counts())
    return input_df

def drop_duplicate_data(input_df):
    input_df.drop_duplicates(subset=['Sentiment', 'Text'], keep='last', inplace=True)
    return input_df

def change_time_format_unix_to_utc(input_df):
    input_df['Time'] = input_df['Time'].apply(lambda x: datetime.datetime.fromtimestamp(x, tz=datetime.timezone.utc))
    return input_df

def preprocess_text(input_df):
    input_df['Text'] = input_df['Text'].apply(strip_html).apply(remove_punctuations).apply(remove_stopwords_and_lemmatize)
    return input_df

def convert_sentiment_to_numerical(input_df):
    sentiment_map = {'Positive' : 1, 'Negetive': 0}
    input_df['Sentiment'] = input_df['Sentiment'].apply(lambda sentiment: sentiment_map[sentiment])
    return input_df

def process_training_data(input_df):
    input_df = create_and_update_sentiment_field(input_df)
    input_data = drop_duplicate_data(input_df)
    input_df = preprocess_text(input_df)
    input_df = change_time_format_unix_to_utc(input_df)
    input_df = convert_sentiment_to_numerical(input_df)
    return input_df

def create_tokenizer(training_text):
    num_tokenizer_words=config.model_config.num_tokenizer_words
    tokenizer = Tokenizer(num_words=num_tokenizer_words)
    tokenizer.fit_on_texts(training_text)
    
    return tokenizer

def get_X_y(input_df):
    return (input_df.Text.values, input_df.Sentiment.values)

def create_and_pad_sequences(token_data):
    pad_token_data = pad_sequences(token_data, 
                                    padding='post', 
                                    maxlen= config.model_config.max_seq_len,
                                    truncating='post')
    return pad_token_data
    
    

stop_word_list = populate_stop_words()
