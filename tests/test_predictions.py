"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import classification_report

from sentimental_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_num_of_predictions = len(sample_input_data[0])

    # When
    result = make_prediction(input_data = sample_input_data[0],do_label = False)

    # Then
    predictions = result.get("predictions")
    #assert isinstance(predictions, np.ndarray)
    #assert isinstance(predictions[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions) == expected_num_of_predictions
    
    print(predictions)
    #_predictions = list(predictions)
    y_true = sample_input_data[1]

    res = classification_report(y_true, predictions, output_dict=True)
  
    assert res['weighted avg']['precision'] > 0.8
    assert res['weighted avg']['recall'] > 0.8
