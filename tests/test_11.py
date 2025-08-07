import pytest
import pickle
from definition_87a4b339efba49eea767321c49542f37 import save_model

def test_save_model_valid_model(tmp_path):
    model = {"function1": lambda x: x * 2, "function2": lambda y: y + 1}
    file_path = tmp_path / "test_model.pkl"
    save_model(model, file_path)
    assert file_path.exists()

    with open(file_path, 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model.keys() == model.keys()

def test_save_model_empty_model(tmp_path):
    model = {}
    file_path = tmp_path / "empty_model.pkl"
    save_model(model, file_path)
    assert file_path.exists()

    with open(file_path, 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model == {}

def test_save_model_non_dict_model(tmp_path):
    model = "This is a string"
    file_path = tmp_path / "string_model.pkl"
    save_model(model, file_path)
    assert file_path.exists()

    with open(file_path, 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model == model

def test_save_model_invalid_file_path():
    model = {"func": lambda x: x}
    file_path = "/invalid/path/model.pkl"
    with pytest.raises(FileNotFoundError):
        save_model(model, file_path)

def test_save_model_none_model(tmp_path):
    model = None
    file_path = tmp_path / "none_model.pkl"
    save_model(model, file_path)
    assert file_path.exists()
    with open(file_path, 'rb') as f:
        loaded_model = pickle.load(f)
    assert loaded_model is None
