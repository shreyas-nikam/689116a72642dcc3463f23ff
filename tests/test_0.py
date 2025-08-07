import pytest
import pandas as pd
from definition_b84de1b919ff482ea1571fe4312ad6d9 import load_raw

def test_load_raw_no_file_path():
    """Test that synthetic data is generated when no file_path is provided."""
    df = load_raw(None)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_load_raw_empty_file(tmp_path):
    """Test that an error is raised for an empty CSV file."""
    file_path = tmp_path / "empty.csv"
    file_path.write_text("")
    with pytest.raises(pd.errors.EmptyDataError):
        load_raw(str(file_path))

def test_load_raw_valid_file(tmp_path):
    """Test loading from a valid CSV file."""
    file_path = tmp_path / "valid.csv"
    data = "loan_id,orig_principal\n1,1000"
    file_path.write_text(data)
    df = load_raw(str(file_path))
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "loan_id" in df.columns
    assert "orig_principal" in df.columns

def test_load_raw_invalid_file_path():
    """Test that FileNotFoundError is raised when file path is invalid"""
    with pytest.raises(FileNotFoundError):
        load_raw("invalid_file_path.csv")

def test_load_raw_correct_types_synthetic():
     """Test the dataframe types are correct when synthetic data is generated"""
     df = load_raw(None)
     assert df['loan_id'].dtype == 'int64'
     assert df['orig_principal'].dtype == 'float64'
     assert df['orig_rate'].dtype == 'float64'
     assert df['orig_term_mths'].dtype == 'int64'

