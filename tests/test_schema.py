import pytest
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError

ml_input_schema = pa.DataFrameSchema({
    "Tenure Months": pa.Column(int, pa.Check.ge(0)), 
    "Monthly Charges": pa.Column(float, pa.Check.ge(0.0)),
    "Total Charges": pa.Column(float, pa.Check.ge(0.0)),
    "CLTV": pa.Column(float, pa.Check.ge(0.0)),
    "Churn Score": pa.Column(float, pa.Check.in_range(0.0, 100.0)) 
})

def test_valid_dataframe_schema():
    """Test that perfectly formatted data passes the schema validation."""
    valid_data = pd.DataFrame({
        "Tenure Months": [12, 24],
        "Monthly Charges": [75.5, 50.0],
        "Total Charges": [906.0, 1200.0],
        "CLTV": [4500.0, 5000.0],
        "Churn Score": [85.0, 40.0]
    })
    
    validated_df = ml_input_schema.validate(valid_data)
    assert not validated_df.empty

def test_invalid_dataframe_schema():
    """Test that corrupted data is caught and throws a SchemaError."""
    invalid_data = pd.DataFrame({
        "Tenure Months": [-10], # Corrupted: Negative tenure
        "Monthly Charges": [75.5],
        "Total Charges": [906.0],
        "CLTV": [4500.0],
        "Churn Score": [150.0] # Corrupted: Score above 100
    })
    
    with pytest.raises(SchemaError):
        ml_input_schema.validate(invalid_data)