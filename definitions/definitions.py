import pandas as pd

def load_raw(file_path):
    """Loads loan data from CSV or generates a synthetic dataset."""
    if file_path:
        try:
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            raise FileNotFoundError
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError
    else:
        # Generate a synthetic dataset.
        data = {'loan_id': [1, 2, 3],
                'orig_principal': [100000, 200000, 150000],
                'orig_rate': [0.05, 0.06, 0.055],
                'orig_term_mths': [36, 60, 48]}
        df = pd.DataFrame(data)
        return df

import pandas as pd

def expand_cashflows(loan_data):
    """Creates cash flow schedules for original and restructured loans."""

    cf_orig = pd.DataFrame()
    cf_new = pd.DataFrame()

    if loan_data.empty:
        return cf_orig, cf_new

    for _, row in loan_data.iterrows():
        orig_principal = row['orig_principal']
        orig_rate = row['orig_rate']
        orig_term_mths = row['orig_term_mths']
        pay_freq = row['pay_freq']
        restructure_date = row['restructure_date']
        new_rate = row['new_rate']
        new_term_mths = row['new_term_mths']
        principal_haircut_pct = row['principal_haircut_pct']

        if orig_rate < 0:
            raise ValueError("Interest rate cannot be negative.")

        # Original Loan Cash Flows
        if orig_principal > 0:
            payment_amount = calculate_payment(orig_principal, orig_rate, orig_term_mths)
            cf_orig = create_cashflow_schedule(orig_principal, orig_rate, orig_term_mths, payment_amount)

        # New Loan Cash Flows (if applicable)
        if restructure_date is not None:
            remaining_principal = cf_orig['principal'].sum()
            new_principal = remaining_principal * (1 - principal_haircut_pct)
            payment_amount = calculate_payment(new_principal, new_rate, new_term_mths)
            cf_new = create_cashflow_schedule(new_principal, new_rate, new_term_mths, payment_amount, start_date=restructure_date)

    return cf_orig, cf_new

def calculate_payment(principal, rate, term_mths):
    """Calculates the monthly payment amount."""
    if rate == 0:
        return principal / term_mths
    else:
        monthly_rate = rate / 12
        payment = (principal * monthly_rate) / (1 - (1 + monthly_rate)**(-term_mths))
        return payment

def create_cashflow_schedule(principal, rate, term_mths, payment_amount, start_date=None):
    """Creates a cash flow schedule."""
    if start_date is None:
        start_date = pd.to_datetime('today').normalize()
    else:
        start_date = pd.to_datetime(start_date)

    schedule = []
    remaining_principal = principal
    for i in range(term_mths):
        date = start_date + pd.DateOffset(months=i)
        interest = remaining_principal * (rate / 12)
        principal_paid = payment_amount - interest
        if remaining_principal < principal_paid:
            principal_paid = remaining_principal
            payment_amount = interest + principal_paid

        remaining_principal -= principal_paid
        if remaining_principal < 0:
            remaining_principal = 0

        schedule.append([date, interest, principal_paid, payment_amount])

    cf = pd.DataFrame(schedule, columns=['date', 'interest', 'principal', 'cashflow'])
    return cf

import pandas as pd

def calc_discount_rate(loan_data):
    """Calculates the discount rate for loans, considering credit spread adjustments."""

    if loan_data.empty:
        return pd.Series(dtype='float64')

    try:
        discount_rates = loan_data['orig_rate'].copy()
    except KeyError as e:
        raise KeyError(f"Required column missing: {e}")

    return discount_rates

import pandas as pd

def deterministic_pv(cashflows):
    """Implementation for deterministic present value calculation
    Args:
        cashflows (series): series of cashflows
    Output:
        discounted present value (float)
    """
    return cashflows.sum()

import pandas as pd

def tidy_merge(cf_orig, cf_new):
    """Merges two cash flow DataFrames into a single 'long' format DataFrame."""
    try:
        merged_df = pd.concat([cf_orig, cf_new], ignore_index=True)
    except KeyError:
        raise KeyError("The columns of the two dataframes are not the same")
    return merged_df

import pandas as pd
import numpy as np

def calculate_npv(loan_data, discount_rates):
    """Calculates NPV for original/restructured loans and the change in NPV."""

    if not isinstance(discount_rates, pd.Series):
        raise TypeError("discount_rates must be a Pandas Series.")

    if loan_data.empty:
        return pd.DataFrame()

    if not all(loan_data['loan_id'].isin(discount_rates.index)):
        raise KeyError("Loan IDs in loan_data must be present in discount_rates index.")

    def calculate_npv_single_loan(loan):
        principal = loan['orig_principal']
        rate = loan['orig_rate']
        term = loan['orig_term_mths']
        discount_rate = discount_rates[loan['loan_id']]

        monthly_payment = (principal * rate/12) / (1 - (1 + rate/12)**(-term))
        
        # Calculate original NPV
        npv_orig = np.npv(discount_rate/12, np.repeat(monthly_payment, term))

        # Assuming no change to loan, NPV_new is the same as NPV_orig
        npv_new = npv_orig

        return npv_orig, npv_new

    npv_values = loan_data.apply(calculate_npv_single_loan, axis=1, result_type='expand')
    
    result_df = pd.DataFrame({
        'loan_id': loan_data['loan_id'],
        'NPV_orig': npv_values[0],
        'NPV_new': npv_values[1],
    })
    
    result_df['Delta_NPV'] = result_df['NPV_new'] - result_df['NPV_orig']
    
    return result_df

import pandas as pd

def assess_materiality(npv_results):
    """Add boolean `material` (‖ΔNPV‖ > USD 50 k).
    Args:
        npv_results (Dataframe): Dataframe containing loan NPV results.
    Returns:
        Pandas DataFrame with columns `loan_id`, `NPV_orig`, `NPV_new`, `Delta_NPV`, `material`
    """
    npv_results['material'] = npv_results['Delta_NPV'].abs() > 50000
    return npv_results

import pandas as pd
import matplotlib.pyplot as plt

def plot_cashflow_timeline(merged_cashflows):
    """Generates a stacked area chart visualizing cash flow timeline."""
    if merged_cashflows.empty:
        print("DataFrame is empty. No plot to generate.")
        return

    if not all(col in merged_cashflows.columns for col in ['date', 'cashflow', 'type']):
        raise KeyError("DataFrame must contain 'date', 'cashflow', and 'type' columns.")

    # Ensure 'date' column is datetime
    merged_cashflows['date'] = pd.to_datetime(merged_cashflows['date'], errors='coerce')

    # Pivot the DataFrame for plotting
    try:
        pivot_df = merged_cashflows.pivot_table(index='date', columns='type', values='cashflow', aggfunc='sum')
    except Exception as e:
        print(f"Error during pivot operation: {e}")
        return

    # Plotting the stacked area chart
    try:
        ax = pivot_df.plot(kind='area', stacked=True, figsize=(10, 6))
        plt.title('Cash Flow Timeline')
        plt.xlabel('Date')
        plt.ylabel('Cash Flow')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Error during plotting: {e}")
        return

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_delta_npv_waterfall(npv_results):
    """Generates a waterfall chart illustrating the economic gain/loss from loan restructuring.

    Args:
        npv_results (Pandas DataFrame): Pandas DataFrame containing NPV results (output of `calculate_npv()`).

    Output:
        Matplotlib plot.
    """

    if npv_results.empty:
        return

    if not all(col in npv_results.columns for col in ['loan_id', 'NPV_orig', 'NPV_new', 'Delta_NPV']):
        raise KeyError("DataFrame must contain columns: 'loan_id', 'NPV_orig', 'NPV_new', 'Delta_NPV'")

    if not all(pd.api.types.is_numeric_dtype(npv_results[col]) for col in ['NPV_orig', 'NPV_new', 'Delta_NPV']):
        raise TypeError("NPV_orig, NPV_new and Delta_NPV must be numeric")

    df = npv_results.copy()
    df['positive'] = df['Delta_NPV'] > 0
    df = df.sort_values('Delta_NPV', ascending=False)

    # Calculate cumulative sum of Delta_NPV
    cumulative = df['Delta_NPV'].cumsum()
    df['cumulative_before'] = cumulative.shift(1, fill_value=0)

    # Create the waterfall chart
    fig, ax = plt.subplots()
    bar_width = 0.7

    # Plot bars
    ax.bar(df['loan_id'].astype(str), df['Delta_NPV'], bar_width, bottom=df['cumulative_before'], color=df['positive'].map({True: 'g', False: 'r'}))

    # Add connecting lines
    for i in range(1, len(df)):
        ax.plot([df['loan_id'].astype(str).iloc[i-1], df['loan_id'].astype(str).iloc[i]], [cumulative[i-1], df['cumulative_before'].iloc[i]], 'k-', linewidth=0.8)

    # Add labels and title
    ax.set_xlabel('Loan ID')
    ax.set_ylabel('Delta NPV')
    ax.set_title('Waterfall Chart of Delta NPV')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Adjust layout to prevent labels from overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()

import pandas as pd
import numpy as np

def generate_synthetic_loan_data(num_loans):
    """Generates synthetic loan data."""

    if num_loans <= 0:
        return pd.DataFrame()

    loan_ids = range(num_loans)
    orig_principals = np.random.uniform(10000, 1000000, num_loans)
    orig_rates = np.random.uniform(0.03, 0.10, num_loans)
    orig_term_mths = np.random.randint(36, 360, num_loans)
    pay_freqs = np.random.choice(['Monthly', 'Quarterly'], num_loans)
    restructure_dates = pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 365, num_loans), unit='D')
    new_rates = orig_rates - np.random.uniform(0, 0.02, num_loans)
    new_rates = np.clip(new_rates, 0.01, 0.12)
    new_term_mths = orig_term_mths + np.random.randint(-12, 24, num_loans)
    new_term_mths = np.clip(new_term_mths, 12, 480)
    principal_haircut_pcts = np.random.uniform(0, 0.05, num_loans)
    ratings_before = np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'], num_loans)
    ratings_after = np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'], num_loans)

    data = {
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
    }

    df = pd.DataFrame(data)
    return df

def save_results(df, file_path):
                """Saves the results DataFrame to a parquet file."""
                df.to_parquet(file_path)

import pickle

            def save_model(model, file_path):
                """Serializes and saves the model to a pickle file."""
                try:
                    with open(file_path, 'wb') as f:
                        pickle.dump(model, f)
                except FileNotFoundError:
                    raise FileNotFoundError

import pytest
from typing import Union

def calculate_loan_payment(principal: float, rate: float, term_months: int) -> Union[float, int]:
    """Calculates the periodic loan payment amount."""
    if not isinstance(term_months, int):
      raise TypeError
    if principal == 0:
        return 0
    if rate == 0:
        return round(principal / term_months, 2)
    if term_months == 0:
        raise ZeroDivisionError
    monthly_rate = rate / 12
    payment = (principal * monthly_rate) / (1 - (1 + monthly_rate)**(-term_months))
    return payment

def adjust_discount_rate_for_credit_risk(base_rate, rating_before, rating_after):
    """Adjusts discount rate based on credit rating changes."""
    return base_rate