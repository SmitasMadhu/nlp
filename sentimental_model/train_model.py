import sys
import os
import contextlib
import logging 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np

from tensorflow import keras,get_logger
#from tensorflow_addons.optimizers import AdamW
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


from sentimental_model.config.core import config
from sentimental_model.model import  create_tf_model
from sentimental_model.processing.data_manager import save_model_data,load_dataset
from sentimental_model.processing.features import create_and_pad_sequences, create_tokenizer, get_X_y, process_training_data


def run_training() -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    data = process_training_data(data)

    X, y = get_X_y(data)

    X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.20, stratify=y, shuffle=True)
    X_train, X_cv, y_train, y_cv = train_test_split(X_t, y_t, test_size=0.20, stratify=y_t, shuffle=True)
    print("Shape of Input  - Train:", X_train.shape)
    print("Shape of Output - Train:", y_train.shape)
    print("Shape of Input  - CV   :", X_cv.shape)
    print("Shape of Output - CV   :", y_cv.shape)
    print("Shape of Input  - Test :", X_test.shape)
    print("Shape of Output - Test :", y_test.shape)


    # Print the sizes of the sets
    print("Train set size:", len(X_train), len(y_train))
    print("Validation set size:", len(X_cv), len(y_cv))
    print("Test set size:", len(X_test), len(y_test))  

    
    tokenizer = create_tokenizer(X_train)

    X_train_tok = tokenizer.texts_to_sequences(X_train)
    X_test_tok = tokenizer.texts_to_sequences(X_test)
    X_cv_tok = tokenizer.texts_to_sequences(X_cv)

    print(X_train_tok[1])
    print(len(X_train_tok))
    
    vocab_size = len(tokenizer.word_index) + 1
    X_train_pad = create_and_pad_sequences(X_train_tok)
    X_test_pad = create_and_pad_sequences(X_test_tok)
    X_cv_pad  = create_and_pad_sequences(X_cv_tok)

    print ('number of unique words in the corpus:', vocab_size)
    print(X_train_pad.shape)
    print(X_train_pad[1])
    
    model = create_tf_model(vocab_size)

    n_epochs = 5
    batchsize = 512

    history = model.fit(X_train_pad, y_train, batch_size=batchsize, epochs=n_epochs, verbose=1, validation_data=(X_cv_pad, y_cv))
    
    
    
    save_model_data(model_to_save=model, tokenizer=tokenizer)    
    y_pred = model.predict(X_test_pad,batch_size=128)
    y_pred_bool = (y_pred>= 0.5)
    print(classification_report(y_test,y_pred_bool)) 
    
    
 
    
   
   
    
if __name__ == "__main__":
    run_training()