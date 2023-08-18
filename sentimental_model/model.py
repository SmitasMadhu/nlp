import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, LSTM, Embedding, Dense,Bidirectional
#from tensorflow_addons.optimizers import AdamW
from sentimental_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR,config


'''
Method for creating TF model. Invoked during the 
pipeline and Hyperparameter tuning. Only exposing 2 parameters
as github takes long time for GridSearch
'''
def create_tf_model(vocab_size):

    EMBEDDING_DIM=32
    maxlen= config.model_config.max_seq_len

    print('Build model... Model M1 ( Embedding -> LSTM -> Output(Sigmoid) )')
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim = EMBEDDING_DIM, input_length=maxlen))
    model.add(LSTM(units=100,  dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    # Try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('Summary of the built model...')
    print(model.summary())       
    
    return model

