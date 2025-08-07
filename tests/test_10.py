import pytest
import pandas as pd
from definition_592367828605467fafc0f4eab0d226c0 import save_results

def test_save_results_empty_df(tmp_path):
    file_path = tmp_path / "test.parquet"
    df = pd.DataFrame()
    save_results(df, str(file_path))
    loaded_df = pd.read_parquet(file_path)
    assert loaded_df.empty

def test_save_results_single_row_df(tmp_path):
    file_path = tmp_path / "test.parquet"
    data = {'col1': [1], 'col2': [2]}
    df = pd.DataFrame(data)
    save_results(df, str(file_path))
    loaded_df = pd.read_parquet(file_path)
    pd.testing.assert_frame_equal(df, loaded_df)

def test_save_results_multiple_rows_df(tmp_path):
    file_path = tmp_path / "test.parquet"
    data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
    df = pd.DataFrame(data)
    save_results(df, str(file_path))
    loaded_df = pd.read_parquet(file_path)
    pd.testing.assert_frame_equal(df, loaded_df)

def test_save_results_different_dtypes(tmp_path):
    file_path = tmp_path / "test.parquet"
    data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c'], 'col3': [1.1, 2.2, 3.3]}
    df = pd.DataFrame(data)
    save_results(df, str(file_path))
    loaded_df = pd.read_parquet(file_path)
    pd.testing.assert_frame_equal(df, loaded_df)

def test_save_results_overwrite_existing_file(tmp_path):
    file_path = tmp_path / "test.parquet"
    # Create dummy file
    file_path.write_text("dummy data")

    data = {'col1': [1], 'col2': [2]}
    df = pd.DataFrame(data)
    save_results(df, str(file_path))
    loaded_df = pd.read_parquet(file_path)
    pd.testing.assert_frame_equal(df, loaded_df)
