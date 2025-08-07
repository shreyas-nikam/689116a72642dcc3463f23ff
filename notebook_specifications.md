
# Jupyter Notebook Specification: Loan Restructuring NPV Calculator

## 1. Notebook Overview

**Learning Goals:**

*   Understand the impact of loan restructurings on Net Present Value (NPV).
*   Learn to calculate NPV for original and restructured loan scenarios using a discounted cash flow (DCF) engine.
*   Visualize cash flow timing differences and the economic impact (gain/loss) of loan restructurings.
*   Explore the sensitivity of NPV to changes in discount rates.

**Expected Outcomes:**

*   Generation of a synthetic dataset of loan characteristics and restructuring scenarios.
*   Implementation of a deterministic DCF engine for calculating NPV of original and restructured loans.
*   Creation of interactive visualizations to illustrate the impact of restructurings.
*   Persistence of model artifacts (NPV engine) and results for subsequent analysis and validation.

## 2. Mathematical and Theoretical Foundations

### 2.1 Net Present Value (NPV)

The Net Present Value (NPV) is a fundamental concept in finance used to determine the present value of a stream of future cash flows, given a specific discount rate. It is calculated using the following formula:

$$ NPV = \sum_{t=0}^{N} \frac{CF_t}{(1 + r)^t} $$

Where:

*   $NPV$ is the Net Present Value.
*   $CF_t$ is the cash flow at time t.
*   $r$ is the discount rate (representing the opportunity cost of capital).
*   $t$ is the time period.
*   $N$ is the total number of time periods.

**Real-world Application:** NPV is widely used in capital budgeting to evaluate the profitability of investments, in project finance to assess the viability of projects, and in loan restructuring to determine the economic impact of modifying loan terms.

### 2.2 Discounted Cash Flow (DCF)

The Discounted Cash Flow (DCF) method is a valuation method used to estimate the value of an investment based on its expected future cash flows. DCF analysis attempts to determine the value of an investment today, based on projections of how much money it will generate in the future.
The core principle of DCF analysis is that an investment is worth the sum of all of its future free cash flows, discounted back to their present value.

### 2.3 Loan Restructuring Impact

Loan restructuring involves modifying the original terms of a loan agreement, such as the interest rate, payment schedule, or principal amount. The NPV of the loan can be significantly affected by restructuring.

**Formulas:**

*   **NPV of Original Loan ($NPV_{orig}$):**

    $$ NPV_{orig} = \sum_{t=1}^{n} \frac{C_{orig,t}}{(1 + r)^{t}} $$

    Where:
    * $C_{orig,t}$ is the cash flow of the original loan at time $t$
    * $r$ is the discount rate (original loan's effective interest rate)
    * $n$ is the number of periods for the original loan

*   **NPV of Restructured Loan ($NPV_{new}$):**

    $$ NPV_{new} = \sum_{t=1}^{m} \frac{C_{new,t}}{(1 + r')^{t}} $$

    Where:
    * $C_{new,t}$ is the cash flow of the restructured loan at time $t$
    * $r'$ is the discount rate for the restructured loan. This could be the original EIR or adjusted for risk.
    * $m$ is the number of periods for the new loan.

*   **Change in NPV ($\Delta NPV$):**

    $$ \Delta NPV = NPV_{new} - NPV_{orig} $$

    A positive $\Delta NPV$ indicates a gain for the borrower (loss for the lender), while a negative $\Delta NPV$ indicates a loss for the borrower (gain for the lender).

## 3. Code Requirements

### 3.1 Libraries

*   `pandas`: Used for data manipulation and analysis, particularly for creating and managing dataframes to store loan data and cash flow schedules.
*   `numpy`: Used for numerical computations, especially for calculating present values and handling arrays of cash flows.
*   `scipy`: Used for financial calculations, such as calculating the present value.
*   `matplotlib`: Used for creating visualizations, including the cash flow timeline and the ΔNPV waterfall chart.
*   `joblib`: Used to save the `npv_engine.pkl`.

### 3.2 Input/Output Expectations

*   **Input:**
    *   Loan data (synthetic generation before loading) containing the following fields:
        *   `loan_id` (int/str): Unique identifier for the loan.
        *   `orig_principal` (float): Original principal amount of the loan.
        *   `orig_rate` (float): Original interest rate of the loan (annual).
        *   `orig_term_mths` (int): Original loan term in months.
        *   `pay_freq` (str): Payment frequency (e.g., "monthly").
        *   `restructure_date` (datetime): Date of the loan restructuring.
        *   `new_rate` (float): New interest rate after restructuring (annual).
        *   `new_term_mths` (int): New loan term in months after restructuring.
        *   `principal_haircut_pct` (float): Percentage of the principal written off during restructuring.
        *   `rating_before` (int): Credit rating before restructuring.
        *   `rating_after` (int): Credit rating after restructuring.
    *   Discount rate (float): Used for calculating the present value of cash flows.

*   **Output:**
    *   `npv_results.parquet`: A parquet file containing the following columns for each loan:
        *   `loan_id` (int/str): Unique identifier for the loan.
        *   `NPV_orig` (float): Net Present Value of the original loan.
        *   `NPV_new` (float): Net Present Value of the restructured loan.
        *   `Delta_NPV` (float): Change in Net Present Value due to restructuring.
        *   `material` (bool): A boolean flag indicating if the absolute value of `Delta_NPV` exceeds a predefined threshold (e.g., USD 50,000).
    *   `npv_engine.pkl`: A pickled object containing the DCF engine functions.

### 3.3 Algorithms and Functions

*   **`load_raw()`:**
    *   Description: Loads the raw loan data from a CSV file or generates a synthetic dataset.
    *   Input: Optional file path to a CSV file.
    *   Output: Pandas DataFrame containing the raw loan data.

*   **`expand_cashflows()`:**
    *   Description: Creates cash flow schedules for both the original and restructured loans.
    *   Input: Pandas DataFrame containing loan data.
    *   Output: Two Pandas DataFrames, `cf_orig` and `cf_new`, each containing the cash flow schedule with columns: `date`, `interest`, `principal`, `cashflow`.

*   **`calc_discount_rate()`:**
    *   Description: Calculates the discount rate to be used for the original and restructured loans. It can consider credit spread adjustments based on rating changes.
    *   Input: Pandas DataFrame containing loan data.
    *   Output: Pandas Series of discount rates for each loan.

*   **`deterministic_pv()`:**
        *   Description: Implementation for deterministic present value calculation
        *   Input: series of cashflows
        *   Output: discounted present value

*   **`tidy_merge()`:**
    *   Description: Merges the original and restructured cash flow schedules into a single "long" format DataFrame.
    *   Input: `cf_orig` and `cf_new` DataFrames.
    *   Output: Pandas DataFrame in long format with columns like `loan_id`, `date`, `cashflow`, `type` (original/restructured).

*   **`calculate_npv()`:**
    *   Description: Calculates the NPV for the original and restructured loans and the change in NPV.
    *   Input: Pandas DataFrame containing loan data and discount rates.
    *   Output: Pandas DataFrame with columns `loan_id`, `NPV_orig`, `NPV_new`, `Delta_NPV`.

*    **`assess_materiality()`:**
    *   Description: Add boolean `material` (‖ΔNPV‖ > USD 50 k).
    *   Input: Dataframe containing loan NPV results.
    *   Output: Pandas DataFrame with columns `loan_id`, `NPV_orig`, `NPV_new`, `Delta_NPV`, `material`

*   **`plot_cashflow_timeline()`:**
    *   Description: Generates a stacked area chart visualizing the cash flow timeline for original and restructured loans.
    *   Input: Pandas DataFrame in long format (output of `tidy_merge()`).
    *   Output: Matplotlib plot.

*   **`plot_delta_npv_waterfall()`:**
    *   Description: Generates a waterfall chart illustrating the economic gain/loss from loan restructuring.
    *   Input: Pandas DataFrame containing NPV results (output of `calculate_npv()`).
    *   Output: Matplotlib plot.

### 3.4 Visualizations

1.  **Cash-Flow Timeline (Stacked Area Chart):**

    *   X-axis: Payment date
    *   Y-axis: Cash flow amount (interest + principal)
    *   Two colored areas: Original vs. restructured loan cash flows.
    *   Purpose: To visually highlight timing shifts and payment burden changes due to restructuring.

2.  **ΔNPV Waterfall Chart:**

    *   Bars representing: `NPV_orig` -> adjustments -> `NPV_new` -> `Delta_NPV`.
    *   Purpose: To make the drivers of economic gain/loss explicit, showing the impact of each adjustment on the NPV.

## 4. Additional Notes or Instructions

*   **Assumptions:**
    *   Deterministic cash flows: No default or prepayment assumptions are included in the model.
    *   Discount rate selection: The original loan's effective interest rate (EIR) is used as the base discount rate, potentially adjusted for changes in credit risk.
    *   Restructuring effectiveness: The restructuring is assumed to be effective immediately from the `restructure_date`.
*   **Constraints:**
    *   The dataset should contain approximately 10 loans.
    *   3-5 loans should be restructured to demonstrate the impact of the model.
    *   Materiality Threshold:  A default of USD 50,000 should be used.
*   **Customization:**
    *   The user should be able to modify the discount rate for sensitivity analysis.
*   **Model Artifact Persistence:**
    *   The `npv_engine.pkl` file should contain all the functions required to run the NPV calculation.
*   **Folder Layout:**
    *   The notebook and all output files (`npv_results.parquet`, `npv_engine.pkl`) should be in a well-organized directory structure.
