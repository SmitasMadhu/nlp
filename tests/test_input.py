
"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from sentimental_model.config.core import config


def test_input_data(sample_input_data):
    # Given
    y_label = sample_input_data[1]
    #Verify that input data contains only 3 labels and we are testing all 3
    assert len(np.unique(y_label,axis=0,return_counts=True)[0]) == 3
    


