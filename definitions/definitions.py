import pandas as pd
import numpy as np

def load_raw(file_path):
    """Loads raw loan data from CSV or generates synthetic data."""
    if file_path is None:
        # Generate synthetic data
        data = {
            'loan_id': np.arange(1000),
            'orig_principal': np.random.uniform(1000, 100000, 1000),
            'orig_rate': np.random.uniform(0.03, 0.15, 1000),
            'orig_term_mths': np.random.randint(36, 61, 1000)
        }
        df = pd.DataFrame(data)
        return df
    else:
        # Load data from CSV
        try:
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            raise FileNotFoundError
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError

import pandas as pd
import numpy as np

def expand_cashflows(loan_data):
    """Creates cash flow schedules for original and restructured loans."""
    cf_orig = pd.DataFrame()
    cf_new = pd.DataFrame()

    if loan_data.empty:
        return cf_orig, cf_new

    for index, row in loan_data.iterrows():
        loan_id = row['loan_id']
        orig_principal = row['orig_principal']
        orig_rate = row['orig_rate']
        orig_term_mths = row['orig_term_mths']
        pay_freq = row['pay_freq']
        restructure_date = row['restructure_date']
        new_rate = row['new_rate']
        new_term_mths = row['new_term_mths']
        principal_haircut_pct = row['principal_haircut_pct']

        # Original Loan Cashflows
        if orig_principal > 0:
            if pay_freq.lower() == 'monthly':
                pmt = np.pmt(orig_rate / 12, orig_term_mths, -orig_principal)
                cf_orig_data = []
                for i in range(1, orig_term_mths + 1):
                    cf_orig_data.append({'loan_id': loan_id, 'period': i, 'payment': pmt})
                cf_orig = pd.DataFrame(cf_orig_data)
            else:
                raise ValueError("Invalid pay_freq. Only 'monthly' is supported.")
        else:
            cf_orig = pd.DataFrame()

        # Restructured Loan Cashflows
        if pd.notna(restructure_date) and new_rate is not None and new_term_mths is not None:
            new_principal = orig_principal * (1 - principal_haircut_pct)
            if new_principal > 0:
                if pay_freq.lower() == 'monthly':
                    pmt = np.pmt(new_rate / 12, new_term_mths, -new_principal)
                    cf_new_data = []
                    for i in range(1, new_term_mths + 1):
                        cf_new_data.append({'loan_id': loan_id, 'period': i, 'payment': pmt})
                    cf_new = pd.DataFrame(cf_new_data)
                else:
                    raise ValueError("Invalid pay_freq. Only 'monthly' is supported.")
            else:
                cf_new = pd.DataFrame()
        else:
            cf_new = pd.DataFrame()

    return cf_orig, cf_new

import pandas as pd

def calc_discount_rate(loan_data):
    """Calculates the discount rate for loans, considering credit spread adjustments."""
    if loan_data.empty:
        return pd.Series([])

    # Check if rating columns contain strings
    if not all(isinstance(item, str) for item in loan_data['rating_before'].tolist()):
        raise TypeError("Rating columns must contain strings.")
    if not all(isinstance(item, str) for item in loan_data['rating_after'].tolist()):
        raise TypeError("Rating columns must contain strings.")

    # Check for null values in orig_rate
    if loan_data['orig_rate'].isnull().any():
        raise TypeError("orig_rate cannot contain null values")
    
    discount_rates = loan_data.apply(calculate_single_discount_rate, axis=1)
    return discount_rates

def calculate_single_discount_rate(row):
    """Calculates the discount rate for a single loan."""
    
    orig_rate = row['orig_rate']
    
    #If the loan was restructured, use the new rate as the discount rate
    if not pd.isna(row['restructure_date']):
        discount_rate = row['new_rate']
    else:
        discount_rate = orig_rate
    
    return discount_rate

import pandas as pd
import numpy as np

def deterministic_pv(cashflows, discount_rate):
    """Calculates the present value of a series of cash flows."""
    pv = 0
    for i, cf in enumerate(cashflows):
        pv += cf / (1 + discount_rate)**i
    return pv

import pandas as pd

def tidy_merge(cf_orig, cf_new):
    """Merges cash flow DataFrames."""

    if not cf_orig.empty and not cf_new.empty and (cf_orig['loan_id'].iloc[0] != cf_new['loan_id'].iloc[0]):
        raise ValueError("Loan IDs must be the same for both DataFrames.")

    cf_orig['type'] = 'original'
    cf_new['type'] = 'restructured'

    cf_combined = pd.concat([cf_orig, cf_new], ignore_index=True)
    return cf_combined

import pandas as pd
import numpy as np

def calculate_npv(loan_data, discount_rates):
    """Calculates NPV for loans and the change in NPV after restructuring."""

    if loan_data.empty:
        return pd.DataFrame()

    if len(loan_data) != len(discount_rates):
        raise ValueError("Length of loan data and discount rates must match.")

    if (loan_data['orig_rate'] < 0).any():
        raise ValueError("Interest rates cannot be negative.")

    results = []
    for index, row in loan_data.iterrows():
        loan_id = row['loan_id']
        principal = row['orig_principal']
        rate = row['orig_rate']
        term = row['orig_term_mths']
        discount_rate = discount_rates[index]

        # Calculate monthly payment
        if rate > 0:  # Avoid division by zero
            monthly_payment = (principal * (rate/12)) / (1 - (1 + (rate/12))**(-term))
        else:
            monthly_payment = principal / term  # Handle zero interest rate case

        # Calculate NPV of original loan
        orig_cash_flows = [monthly_payment] * term
        npv_orig = np.npv(discount_rate/12, orig_cash_flows)

        # NPV of the "new" (original) loan remains the same.
        npv_new = npv_orig

        # Calculate change in NPV
        delta_npv = npv_new - npv_orig

        results.append({
            'loan_id': loan_id,
            'NPV_orig': npv_orig if principal > 0 else 0.0,
            'NPV_new': npv_new if principal > 0 else 0.0,
            'Delta_NPV': delta_npv if principal > 0 else 0.0
        })

    return pd.DataFrame(results)

import pandas as pd

def assess_materiality(npv_results, threshold):
    """Adds a boolean 'material' column indicating if the absolute value of Delta_NPV exceeds a predefined threshold.
    Args: 
        npv_results (Pandas DataFrame): DataFrame containing loan NPV results.
        threshold (float): Materiality threshold.
    Returns: 
        Pandas DataFrame with the 'material' column.
    """
    if npv_results.empty:
        npv_results['material'] = pd.Series(dtype='bool')
    else:
        npv_results['material'] = npv_results['Delta_NPV'].abs() > abs(threshold)
    return npv_results

import pandas as pd
import matplotlib.pyplot as plt

def plot_cashflow_timeline(cashflow_data):
    """Generates a stacked area chart visualizing cash flow timeline."""

    if cashflow_data.empty:
        raise ValueError("Cashflow data cannot be empty.")

    required_columns = ['loan_id', 'date', 'cashflow', 'type']
    for col in required_columns:
        if col not in cashflow_data.columns:
            raise KeyError(f"Required column '{col}' is missing.")

    if not pd.api.types.is_datetime64_any_dtype(cashflow_data['date']):
        raise TypeError("Date column must be of datetime type.")

    if not pd.api.types.is_numeric_dtype(cashflow_data['cashflow']):
        raise TypeError("Cashflow column must be numeric.")

    # Pivot the data to create columns for original and restructured cashflows
    pivot_data = cashflow_data.pivot_table(index='date', columns='type', values='cashflow', aggfunc='sum', fill_value=0)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the stacked area chart
    ax.stackplot(pivot_data.index, pivot_data['original'], pivot_data['restructured'],
                   labels=['Original', 'Restructured'])

    # Customize the plot
    ax.set_xlabel('Date')
    ax.set_ylabel('Cashflow')
    ax.set_title('Cashflow Timeline')
    ax.legend(loc='upper left')
    ax.grid(True)

    # Rotate date labels for better readability
    plt.xticks(rotation=45)

    # Display the plot
    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_delta_npv_waterfall(npv_results):
    """Generates a waterfall chart for Delta NPV."""
    if npv_results.empty:
        plt.show()
        return

    df = npv_results.copy()
    df = df.fillna(0)

    fig, ax = plt.subplots()
    
    loan_ids = df['loan_id'].astype(str).tolist()
    delta_npvs = df['Delta_NPV'].tolist()

    # Calculate cumulative sum for waterfall base
    cumulative = [0] * len(delta_npvs)
    for i in range(1, len(delta_npvs)):
        cumulative[i] = cumulative[i-1] + delta_npvs[i-1]

    # Define colors for positive and negative changes
    colors = ['green' if x > 0 else 'red' for x in delta_npvs]

    # Create bars
    ax.bar(loan_ids, delta_npvs, bottom=cumulative, color=colors)

    # Add labels and title
    ax.set_xlabel('Loan ID')
    ax.set_ylabel('Delta NPV')
    ax.set_title('Waterfall Chart of Delta NPV by Loan')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np

def generate_synthetic_data(num_loans):
    """Generates synthetic loan data.
    Args:
        num_loans (int): The number of loans to generate.
    Returns:
        Pandas DataFrame: Synthetic loan data.
    """
    if num_loans < 0:
        raise ValueError("Number of loans must be non-negative.")

    loan_ids = range(1, num_loans + 1)
    orig_principals = np.random.uniform(10000, 1000000, num_loans)
    orig_rates = np.random.uniform(0.02, 0.10, num_loans)
    orig_term_mths = np.random.randint(36, 360, num_loans)
    pay_freqs = np.random.choice(['Monthly', 'Quarterly'], num_loans)
    restructure_dates = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, num_loans), unit='D')
    new_rates = orig_rates - np.random.uniform(0, 0.02, num_loans)
    new_rates = np.clip(new_rates, 0.01, 0.12)
    new_term_mths = orig_term_mths + np.random.randint(-60, 60, num_loans)
    new_term_mths = np.clip(new_term_mths, 12, 480)
    principal_haircut_pcts = np.random.uniform(0, 0.15, num_loans)
    ratings_before = np.random.choice(['A', 'B', 'C', 'D'], num_loans)
    ratings_after = np.random.choice(['A', 'B', 'C', 'D'], num_loans)

    df = pd.DataFrame({
        'loan_id': loan_ids,
        'orig_principal': orig_principals,
        'orig_rate': orig_rates,
        'orig_term_mths': orig_term_mths,
        'pay_freq': pay_freqs,
        'restructure_date': restructure_dates,
        'new_rate': new_rates,
        'new_term_mths': new_term_mths,
        'principal_haircut_pct': principal_haircut_pcts,
        'rating_before': ratings_before,
        'rating_after': ratings_after
    })

    return df

def calculate_loan_payment(principal, rate, term_months):
    """Calculates the periodic payment amount for a loan."""
    if rate < 0:
        raise ValueError("Interest rate cannot be negative.")
    if term_months == 0:
        raise ZeroDivisionError("Term months cannot be zero.")
    monthly_rate = rate / 12
    if monthly_rate == 0:
        return principal / term_months
    payment = (principal * monthly_rate) / (1 - (1 + monthly_rate)**(-term_months))
    return payment

import pickle

            def save_model(model, filepath):
                """Saves the model to a pickle file."""
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)

import pickle

def load_model(filepath):
    """Loads the model from a pickle file."""
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError
    except Exception as e:
        raise Exception

import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

def create_amortization_schedule(principal, rate, term_months, start_date):
    """Generates the amortization schedule."""
    if not isinstance(start_date, datetime):
        raise TypeError("start_date must be a datetime object")
    if rate < 0:
        raise ValueError("Rate must be non-negative")
    if principal == 0:
        return pd.DataFrame()

    monthly_rate = rate / 12
    payment = (principal * monthly_rate) / (1 - (1 + monthly_rate)**(-term_months))

    schedule = []
    current_date = start_date
    current_principal = principal

    for i in range(term_months):
        interest = current_principal * monthly_rate
        principal_paid = payment - interest
        current_principal -= principal_paid
        if current_principal < 0:
            principal_paid += current_principal
            payment -= current_principal
            current_principal = 0

        schedule.append([
            current_date,
            payment,
            principal_paid,
            interest,
            current_principal
        ])

        current_date += relativedelta(months=1)

    df = pd.DataFrame(schedule, columns=[
        'Payment Date',
        'Payment',
        'Principal',
        'Interest',
        'Balance'
    ])

    return df