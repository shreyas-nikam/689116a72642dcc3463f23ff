import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import streamlit as st


def generate_enhanced_loan_data_adaptable(num_loans_param):
    """
    Generate enhanced synthetic loan data with more realistic and diverse scenarios.
    Modified to accept num_loans_param.
    """
    data = []
    num_loans = num_loans_param

    loan_ids = [f"L{i:03d}" for i in range(1, num_loans + 1)]
    issuance_dates = [datetime(2022, random.randint(1, 12), random.randint(1, 28)) for _ in range(num_loans)]
    original_amounts = [round(random.uniform(50000, 1000000), 2) for _ in range(num_loans)]
    original_terms = [random.choice([12, 24, 36, 48, 60, 72, 84, 96, 108, 120]) for _ in range(num_loans)] # in months
    original_rates = [round(random.uniform(0.03, 0.12), 4) for _ in range(num_loans)]
    credit_scores = [random.randint(550, 850) for _ in range(num_loans)]
    collateral_values = [round(amount * random.uniform(0.8, 1.5), 2) for amount in original_amounts]
    loan_types = random.choices(["Mortgage", "Auto", "Personal", "Business"], k=num_loans)
    regions = random.choices(["North", "South", "East", "West", "Central"], k=num_loans)

    # Restructuring parameters
    restructured_indices = random.sample(range(num_loans), k=int(num_loans * random.uniform(0.4, 0.7)))
    restructure_reasons = [
        "Economic Hardship", "Market Conditions", "Credit Deterioration",
        "Strategic Re-alignment", "Interest Rate Optimization"
    ]

    for i in range(num_loans):
        is_restructured = i in restructured_indices
        restructure_date = None
        new_term_mths = None
        new_rate = None
        reason = None
        new_payment_freq = None

        if is_restructured:
            # Restructure date must be after issuance date and before original term ends
            max_restructure_date = issuance_dates[i] + timedelta(days=original_terms[i] * 30 / 2) # Arbitrary mid-term
            restructure_date = issuance_dates[i] + timedelta(days=random.randint(30, (max_restructure_date - issuance_dates[i]).days))

            new_term_mths = original_terms[i] + random.choice([12, 24, 36]) # Extend term
            new_rate = round(original_rates[i] + random.uniform(-0.01, 0.01), 4) # Adjust rate slightly
            reason = random.choice(restructure_reasons)
            new_payment_freq = random.choice(["Monthly", "Quarterly"])
        else:
            new_payment_freq = random.choice(["Monthly", "Quarterly"])
            
        data.append({
            "loan_id": loan_ids[i],
            "issuance_date": issuance_dates[i],
            "original_amount": original_amounts[i],
            "original_term_mths": original_terms[i],
            "original_rate": original_rates[i],
            "credit_score": credit_scores[i],
            "collateral_value": collateral_values[i],
            "loan_type": loan_types[i],
            "region": regions[i],
            "is_restructured": is_restructured,
            "restructure_date": restructure_date,
            "new_term_mths": new_term_mths,
            "new_rate": new_rate,
            "restructure_reason": reason,
            "original_payment_freq": random.choice(["Monthly", "Quarterly"]),
            "new_payment_freq": new_payment_freq
        })
    return data


def _calculate_payment_schedule_params(annual_rate, term_mths, pay_freq, principal_amount):
    """Calculates effective rate, number of payments, and periodic payment."""
    if pay_freq == "Monthly":
        n_payments = term_mths
        effective_rate = annual_rate / 12
    elif pay_freq == "Quarterly":
        n_payments = term_mths // 3
        effective_rate = annual_rate / 4
    else:
        raise ValueError("Invalid payment frequency")

    if effective_rate > 0:
        periodic_payment = principal_amount * (effective_rate * (1 + effective_rate)**n_payments) / ((1 + effective_rate)**n_payments - 1)
    else:
        periodic_payment = principal_amount / n_payments # Simple interest for 0 rate
    
    return effective_rate, n_payments, periodic_payment


def _calculate_amortization_schedule(loan_id, principal_amount, annual_rate, term_mths, pay_freq, start_date):
    """Calculates the amortization schedule for a single loan."""
    effective_rate, n_payments, periodic_payment = _calculate_payment_schedule_params(annual_rate, term_mths, pay_freq, principal_amount)

    schedule = []
    remaining_balance = principal_amount
    current_date = start_date

    for i in range(1, n_payments + 1):
        interest_payment = remaining_balance * effective_rate
        principal_paid = periodic_payment - interest_payment
        
        # Adjust for last payment to clear balance
        if i == n_payments:
            principal_paid = remaining_balance # Ensure last payment clears balance
            periodic_payment = interest_payment + principal_paid
            
        remaining_balance -= principal_paid

        if pay_freq == "Monthly":
            current_date = start_date + pd.DateOffset(months=i)
        elif pay_freq == "Quarterly":
            current_date = start_date + pd.DateOffset(months=i*3)

        schedule.append({
            "loan_id": loan_id,
            "payment_number": i,
            "date": current_date,
            "principal_payment": principal_paid,
            "interest_payment": interest_payment,
            "total_payment": periodic_payment,
            "remaining_balance": max(0, remaining_balance) # Ensure balance doesn't go negative
        })

    return pd.DataFrame(schedule)


def expand_cashflows(df_loans):
    """Expands original and new loan terms into detailed cash flow schedules."""
    all_cf_orig = []
    all_cf_new = []

    for _, row in df_loans.iterrows():
        # Original Loan Cash Flows
        orig_schedule = _calculate_amortization_schedule(
            row["loan_id"],
            row["original_amount"],
            row["original_rate"],
            row["original_term_mths"],
            row["original_payment_freq"],
            row["issuance_date"]
        )
        orig_schedule["scenario"] = "Original"
        all_cf_orig.append(orig_schedule)

        # New Loan Cash Flows (if restructured)
        if row["is_restructured"]:
            # Calculate remaining balance at restructure date for the original loan
            # This is a simplification; a real system would calculate the balance precisely
            # based on payments made up to the restructure date.
            # For this simulation, we'll assume the new loan amount is the original principal
            # (or a slightly adjusted one, but sticking to original for simplicity).
            # A more robust approach would compute the original loan's outstanding balance at restructure_date.
            
            # Let's find the last payment date before restructure to get the principal remaining
            original_remaining_principal_at_restructure = row["original_amount"] # Simplified, assuming a new start for the new loan
            if row["restructure_date"]:
                 # Re-calculate original schedule to find remaining balance at restructure date.
                 # This is computationally intensive. For simplicity, we'll assume new loan is on original amount
                 # or a fixed percentage of it at restructure for the demo, or just use the original amount as basis.
                 # For the purpose of this lab, we can consider the "new_loan" starts with its new amount.
                 # Or, find the balance at restructure_date from the original schedule.

                temp_orig_schedule_upto_restructure = _calculate_amortization_schedule(
                    row["loan_id"],
                    row["original_amount"],
                    row["original_rate"],
                    row["original_term_mths"],
                    row["original_payment_freq"],
                    row["issuance_date"]
                )
                # Find the balance at or just after restructure_date
                relevant_rows = temp_orig_schedule_upto_restructure[temp_orig_schedule_upto_restructure['date'] <= row['restructure_date']]
                if not relevant_rows.empty:
                    original_remaining_principal_at_restructure = relevant_rows.iloc[-1]['remaining_balance']
                else:
                    original_remaining_principal_at_restructure = row['original_amount'] # If restructure date is before first payment

            new_schedule = _calculate_amortization_schedule(
                row["loan_id"],
                original_remaining_principal_at_restructure, # Principal for the new loan starts from here
                row["new_rate"],
                row["new_term_mths"],
                row["new_payment_freq"],
                row["restructure_date"]
            )
            new_schedule["scenario"] = "New"
            all_cf_new.append(new_schedule)
        else:
            # If not restructured, new cash flows are identical to original
            orig_schedule_copy = orig_schedule.copy()
            orig_schedule_copy["scenario"] = "New" # Mark as "New" for non-restructured to compare apples-to-apples
            all_cf_new.append(orig_schedule_copy)

    cf_orig_df = pd.concat(all_cf_orig).reset_index(drop=True) if all_cf_orig else pd.DataFrame()
    cf_new_df = pd.concat(all_cf_new).reset_index(drop=True) if all_cf_new else pd.DataFrame()

    return cf_orig_df, cf_new_df


def calc_discount_rate(df_loans):
    """Calculates effective discount rates for original and new scenarios based on credit score."""
    # This is a simplified model. In reality, discount rates are more complex (e.g., yield curves, credit spreads).
    # Assume a base rate and add a spread based on credit score.
    base_rate = 0.04 # Example base rate
    
    # Map credit score to a spread - lower score means higher spread
    def get_credit_spread(score):
        if score >= 750: return 0.005 # Excellent
        elif score >= 700: return 0.010 # Very Good
        elif score >= 650: return 0.015 # Good
        elif score >= 600: return 0.020 # Fair
        else: return 0.025 # Poor

    df_loans["original_discount_rate"] = df_loans.apply(lambda row: base_rate + get_credit_spread(row["credit_score"]), axis=1)
    
    # For new loans, adjust based on new_rate if available, otherwise use original logic with potential adjustment
    df_loans["new_discount_rate"] = df_loans.apply(
        lambda row: (base_rate + get_credit_spread(row["credit_score"]) + (row["new_rate"] - row["original_rate"]) 
                     if row["is_restructured"] else row["original_discount_rate"]),
        axis=1
    )
    return df_loans


def tidy_merge(cf_orig, cf_new, df_loans_with_rates):
    """Merges cash flow data with loan details and prepares for NPV calculation."""
    if cf_orig.empty or cf_new.empty:
        return pd.DataFrame()

    # Ensure 