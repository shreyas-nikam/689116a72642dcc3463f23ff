import pytest
import pandas as pd
from datetime import datetime
from definition_9694073c924f46fcb571a983a5b5eb80 import create_amortization_schedule

def is_close_to_expected(actual: pd.DataFrame, expected: pd.DataFrame):
    try:
        pd.testing.assert_frame_equal(actual, expected, atol=1e-5)
        return True
    except AssertionError:
        return False

def test_empty_schedule():
    principal = 0
    rate = 0.05
    term_months = 12
    start_date = datetime(2024, 1, 1)
    schedule = create_amortization_schedule(principal, rate, term_months, start_date)
    assert isinstance(schedule, pd.DataFrame)
    assert len(schedule) == 0

def test_invalid_rate():
    with pytest.raises(ValueError):
        create_amortization_schedule(10000, -0.05, 12, datetime(2024, 1, 1))

def test_short_term_loan():
    principal = 1000
    rate = 0.12
    term_months = 3
    start_date = datetime(2024, 1, 1)
    schedule = create_amortization_schedule(principal, rate, term_months, start_date)
    assert isinstance(schedule, pd.DataFrame)
    assert len(schedule) == 3

def test_normal_loan():
    principal = 100000
    rate = 0.05
    term_months = 60
    start_date = datetime(2024, 1, 1)
    schedule = create_amortization_schedule(principal, rate, term_months, start_date)
    assert isinstance(schedule, pd.DataFrame)
    assert len(schedule) == 60

def test_non_datetime_start_date():
    principal = 1000
    rate = 0.12
    term_months = 3
    start_date = "2024-01-01"
    with pytest.raises(TypeError):
        create_amortization_schedule(principal, rate, term_months, start_date)
