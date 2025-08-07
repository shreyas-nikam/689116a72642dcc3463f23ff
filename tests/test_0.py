import pytest
import pandas as pd
from definition_aeea39366bb74beb9fc5e4cf251df275 import load_raw

def test_load_raw_no_file_path():
    """Test that load_raw() returns a DataFrame when no file_path is provided (synthetic data)."""
    df = load_raw(None)
    assert isinstance(df, pd.DataFrame)

def test_load_raw_invalid_file_path():
    """Test that load_raw() handles an invalid file path gracefully."""
    with pytest.raises(FileNotFoundError):
        load_raw("nonexistent_file.csv")

def test_load_raw_empty_csv(tmp_path):
    """Test that load_raw() handles an empty CSV file."""
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "empty.csv"
    p.write_text("")
    with pytest.raises(pd.errors.EmptyDataError):
        load_raw(str(p))

def test_load_raw_valid_csv(tmp_path):
    """Test that load_raw() correctly loads data from a valid CSV file."""
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.csv"
    csv_data = "loan_id,orig_principal,orig_rate,orig_term_mths\n1,100000,0.05,36"
    p.write_text(csv_data)
    df = load_raw(str(p))
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.iloc[0]['loan_id'] == 1
    assert df.iloc[0]['orig_principal'] == 100000
    assert df.iloc[0]['orig_rate'] == 0.05
    assert df.iloc[0]['orig_term_mths'] == 36

def test_load_raw_column_types(tmp_path):
    """Test that load_raw() infers correct column types from a CSV file."""
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "types.csv"
    csv_data = "loan_id,orig_principal,orig_rate,orig_term_mths\n1,100000.0,0.05,36"
    p.write_text(csv_data)
    df = load_raw(str(p))
    assert isinstance(df['loan_id'].iloc[0], int)
    assert isinstance(df['orig_principal'].iloc[0], float)
    assert isinstance(df['orig_rate'].iloc[0], float)
    assert isinstance(df['orig_term_mths'].iloc[0], int)

