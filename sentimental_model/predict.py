import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import pandas as pd
import numpy as np
import tensorflow as tf
#from tensorflow_addons.optimizers import AdamW
from az_senti_model import __version__ as _version
from az_senti_model.config.core import config

from az_senti_model.processing.data_manager import load_model_data
from az_senti_model.processing.data_manager import pre_pipeline_preparation
from az_senti_model.processing.validation import validate_inputs
from az_senti_model.processing.features import preprocess_text,create_and_pad_sequences
logging.basicConfig(filename='std2.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',level=logging.CRITICAL)
logging.getLogger().addHandler(logging.FileHandler('std2.log'))
tf.keras.utils.disable_interactive_logging()

model_file_name = f"{config.app_config.model_save_file}{_version}.keras"
tokenizer_file_name = f"{config.app_config.tokenizer_save_file}{_version}.json"
az_senti_model, tokenizer = load_model_data(model_file_name=model_file_name,
                                 tokenizer_file_name=tokenizer_file_name)


def make_prediction(*,input_data: dict, return_as_tuple=False) -> dict:
    '''
    Make a prediction using a saved model
    the output format is either np arrray of the target label
    or a string representation of the same
    ''' 
       
    results = {"predictions": None, "version": _version, }
    try :
        validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
        #return results
        if errors:
            results['errors'] = errors
        else:
            data = preprocess_text(validated_data)
            sequences = create_and_pad_sequences(tokenizer=tokenizer, data=data.Text.values)
            predictions = az_senti_model.predict(sequences)
            
            out_list = []
            for element in predictions:
                out_list.append( element[0] >= 0.5 if return_as_tuple == False else (element[0] >= 0.5, element[0]))
         
            results["predictions"] = out_list
                
        print(results)
    except Exception as e:
        print("exception ", e)
        results['errors'] = str(e)
        
    return results

if __name__ == "__main__":
    
    data_in = {'Id': [1], 'ProductId': ['B001E4KFG0'], 'UserId': ['A3SGXH7AUHU8GW'], 'ProfileName': ['delmartian'], 'HelpfulnessNumerator': [1], 'HelpfulnessDenominator': [1], 'Score': [5], 'Time': [1303862400], 'Summary': ['Good Quality Dog Food'], 'Text': ['I have bought several of the Vitality canned dog food products and have found them all to be of good quality. The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most.']}
    make_prediction(input_data=data_in, return_as_tuple= True)

