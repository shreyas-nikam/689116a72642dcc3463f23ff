import pytest
from definition_3caa953805b04d51a775b13ba1519708 import save_model
import pickle
import os

def test_save_model_success(tmp_path):
    model = {"key": "value"}
    filepath = tmp_path / "test_model.pkl"
    save_model(model, str(filepath))
    assert os.path.exists(filepath)
    with open(filepath, 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model == model

def test_save_model_empty_model(tmp_path):
    model = {}
    filepath = tmp_path / "test_model.pkl"
    save_model(model, str(filepath))
    assert os.path.exists(filepath)
    with open(filepath, 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model == model

def test_save_model_complex_object(tmp_path):
    class MyClass:
        def __init__(self, x):
            self.x = x
    model = MyClass(5)
    filepath = tmp_path / "test_model.pkl"
    save_model(model, str(filepath))
    assert os.path.exists(filepath)
    with open(filepath, 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model.x == model.x

def test_save_model_filepath_as_path_object(tmp_path):
    model = {"key": "value"}
    filepath = tmp_path / "test_model.pkl"
    save_model(model, filepath) # Pass Path object directly
    assert os.path.exists(filepath)
    with open(filepath, 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model == model

def test_save_model_no_overwrite(tmp_path):
    model = {"key": "value"}
    filepath = tmp_path / "test_model.pkl"
    # Create a dummy file first
    with open(filepath, 'w') as f:
        f.write("Dummy Data")

    save_model(model, str(filepath))
    assert os.path.exists(filepath)

    with open(filepath, 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model == model
