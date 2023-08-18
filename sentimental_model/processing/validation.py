import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import sentimental_model
import typing as t
from typing import List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from sentimental_model.config.core import config
from sentimental_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(data_frame = input_df)
    validated_data = pre_processed[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs = validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    Id: Optional[int] 
    ProductId:  Optional[str]
    UserId: Optional[str]
    ProfileName: Optional[str]
    HelpfulnessNumerator: Optional[int]
    HelpfulnessDenominator: Optional[int]
    Score: Optional[int]
    Time: Optional[int]
    Summary: Optional[str]
    Text: Optional[str]
    
    


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]