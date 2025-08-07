
# Jupyter Notebook Specification: Loan Restructuring Impact Analyzer

## 1. Notebook Overview

This Jupyter Notebook provides a focused environment for financial analysts and students to evaluate the economic impact of loan restructuring events. It implements a deterministic Net Present Value (NPV) modeling framework to compare the present value of cash flows under original and revised loan terms.

### Learning Goals

Upon completion of this notebook, users will be able to:
*   Create a synthetic dataset representing approximately 10 loans, including all necessary fields for a comprehensive NPV study of loan restructuring.
*   Develop a single deterministic discounted-cash-flow engine capable of computing $NPV_{orig}$, $NPV_{new}$, and $\Delta NPV$ for each loan within the dataset.
*   Interpret the results effectively through two key visualizations: one depicting cash-flow timing differences and another illustrating the economic gain/loss via a waterfall chart.
*   Persist essential model artifacts, including the NPV engine (`npv_engine.pkl`) and the calculated NPV results (`npv_results.parquet`), to facilitate seamless subsequent validation and governance processes.

### Expected Outcomes

*   A well-structured Jupyter Notebook with clear markdown explanations and distinct code sections.
*   A synthetic loan dataset (approximately 10 rows) generated and pre-processed.
*   Computed $NPV_{orig}$, $NPV_{new}$, and $\Delta NPV$ for each loan.
*   Visualizations showcasing the impact of restructuring on cash flow timelines and the drivers of $\Delta NPV$.
*   Serialized model engine and computed results files.

## 2. Mathematical and Theoretical Foundations

### Net Present Value (NPV)

Net Present Value (NPV) is a core concept in finance, used to evaluate the profitability of an investment or project. It calculates the present value of future cash flows, discounted at a specific rate, and subtracts the initial investment (though for loan restructuring, we are comparing two streams of cash flows). For loan restructuring, NPV helps quantify the economic gain or loss from modifying loan terms.

The fundamental formula for Net Present Value (NPV) is given by:

$$NPV = \sum_{t=1}^{T} \frac{CF_t}{(1+r)^t}$$

Where:
*   $CF_t$ represents the cash flow (interest plus principal payments) at time $t$.
*   $t$ is the time period in which the cash flow occurs (e.g., month, quarter, year).
*   $r$ is the discount rate per period, reflecting the time value of money and the risk associated with the cash flows.
*   $T$ is the total number of time periods over which cash flows are projected.

### Economic Gain/Loss ($\Delta NPV$)

The economic impact of a loan restructuring is measured by the change in NPV. This is calculated as the difference between the NPV of the new (restructured) loan terms and the NPV of the original loan terms.

$$\Delta NPV = NPV_{new} - NPV_{orig}$$

*   A positive $\Delta NPV$ indicates an economic gain for the lender (the present value of the new cash flows is higher than the original).
*   A negative $\Delta NPV$ indicates an economic loss for the lender (the present value of the new cash flows is lower than the original).

### Discount Rate Determination

The choice of discount rate $r$ is critical. In this analysis:
*   For the original loan cash flows, the discount rate ($r_{orig}$) should typically be the loan's original Effective Interest Rate (EIR). This approach isolates the impact of changes in cash flow timing and amount due to restructuring.
*   For the new (restructured) loan cash flows, the discount rate ($r_{new}$) starts with the original EIR but can be adjusted to reflect changes in the borrower's credit risk. If the borrower's credit rating worsens (indicated by `rating_after` being lower than `rating_before`), an additional credit spread (e.g., 100 basis points or 0.01) is added to the original EIR to reflect the higher required return for the increased risk.
    $$r_{new} = r_{orig} + \text{Credit Spread Adjustment}$$
    This adjustment ensures that the NPV calculation for the restructured loan accounts for both the new terms and any change in perceived credit quality.

### Cash Flow Generation

Cash flows ($CF_t$) for both original and restructured loans will be derived from standard loan amortization schedules. These schedules will incorporate:
*   The original principal, interest rate, and term for $CF_{orig}$.
*   The new principal (adjusted for `principal_haircut_pct`), new interest rate (`new_rate`), and new term (`new_term_mths`) for $CF_{new}$.
*   Payments are typically monthly or based on the `pay_freq` field.

## 3. Code Requirements

This section outlines the logical flow and functional requirements for the Jupyter Notebook's code sections. No actual Python code will be provided.

### Expected Libraries

The following Python libraries are expected to be used:
*   `pandas` for data manipulation and DataFrame operations.
*   `numpy` for numerical operations, especially financial calculations.
*   `scipy` for potentially more advanced financial functions (e.g., present value, future value functions if not implemented manually).
*   `matplotlib` for generating static visualizations.

### Input/Output Expectations

#### Input Data
The notebook will begin by generating a synthetic dataset representing approximately 10 loan contracts. This dataset will include the following fields:
*   `loan_id`: Unique identifier for each loan.
*   `orig_principal`: Original principal amount of the loan.
*   `orig_rate`: Original annual interest rate.
*   `orig_term_mths`: Original loan term in months.
*   `pay_freq`: Payment frequency (e.g., 'Monthly', 'Quarterly').
*   `restructure_date`: Date of loan restructuring (for restructured loans).
*   `new_rate`: New annual interest rate post-restructuring.
*   `new_term_mths`: New loan term in months post-restructuring.
*   `principal_haircut_pct`: Percentage of principal forgiven/cut.
*   `rating_before`: Borrower's credit rating before restructuring (e.g., 1-5 scale).
*   `rating_after`: Borrower's credit rating after restructuring (e.g., 1-5 scale).

Note: 3-5 loans in the synthetic dataset should have restructuring events; others should remain unchanged to allow for comparison.

#### Intermediate Data
*   `loan_cf_master.parquet`: A processed DataFrame containing detailed cash flow schedules for both original and restructured loans, including `date`, `interest`, `principal`, and `cashflow` columns for each loan-date combination.

#### Output Data
*   `npv_engine.pkl`: A serialized file containing the core NPV calculation engine/functions.
*   `npv_results.parquet`: A DataFrame saved to a Parquet file, containing per-loan outputs with the following columns: `loan_id`, `NPV_orig`, `NPV_new`, `Delta_NPV`, and `material` (boolean flag).

### Algorithms or Functions to be Implemented

The notebook's logical flow will follow these distinct code sections:

#### Section 1: Data Generation and Pre-processing

*   **1.1 Load Raw Data / Generate Synthetic Data**
    *   **Function:** `load_raw()`
    *   **Description:** This function will generate a `pandas.DataFrame` representing the toy loan portfolio (approx. 10 rows). It will populate all required input fields (`loan_id` through `rating_after`) with realistic synthetic values. For non-restructured loans, `restructure_date`, `new_rate`, `new_term_mths`, and `principal_haircut_pct` will be set to indicate no change (e.g., `None` or original values).
*   **1.2 Expand Cashflows**
    *   **Function:** `expand_cashflows(df_loans)`
    *   **Description:** Takes the raw loan DataFrame and generates detailed amortization tables for both the original and (if applicable) restructured terms for each loan. This involves calculating periodic interest and principal payments. It will produce two derived tables, `cf_orig` and `cf_new`, which contain `date`, `interest`, `principal`, and `cashflow` (interest + principal) for each payment period. `principal_haircut_pct` will be applied to adjust the principal for `cf_new`.
*   **1.3 Calculate Discount Rate**
    *   **Function:** `calc_discount_rate(df_loans)`
    *   **Description:** For each loan, it will determine the appropriate discount rate (`r_orig` and `r_new`). `r_orig` will be set to `orig_rate`. `r_new` will be `orig_rate` unless `rating_after` indicates a worsened credit rating compared to `rating_before`, in which case a predefined spread (e.g., 100 basis points or 0.01) will be added to `orig_rate`.
*   **1.4 Tidy and Merge Data**
    *   **Function:** `tidy_merge(cf_orig, cf_new, df_loans)`
    *   **Description:** Combines the original and new cash flow tables (`cf_orig`, `cf_new`) with the loan metadata (`df_loans`) into a single, long-format DataFrame (`loan_cf_master`). This DataFrame will be structured with columns such as `loan_id`, `date`, `cashflow_orig`, `cashflow_new`, `discount_rate_orig`, `discount_rate_new`, etc., suitable for NPV calculation.
*   **1.5 Save Processed Data**
    *   **Description:** The `loan_cf_master` DataFrame will be saved to `loan_cf_master.parquet` for later use.

#### Section 2: NPV Calculation Engine

*   **2.1 NPV Calculation Function**
    *   **Function:** `calculate_npv(cashflows, discount_rate)`
    *   **Description:** A core function that takes a series of periodic cash flows and a corresponding periodic discount rate. It calculates the Net Present Value using the formula $NPV = \sum \frac{CF_t}{(1+r)^t}$. This function will be applied iteratively for each loan's original and new cash flow streams.
*   **2.2 Run NPV Analysis**
    *   **Function:** `run_npv_analysis(df_master)`
    *   **Description:** Orchestrates the NPV calculations for all loans. For each `loan_id` in the `df_master` DataFrame, it will:
        *   Extract `cashflow_orig` and `discount_rate_orig` to calculate `NPV_orig`.
        *   Extract `cashflow_new` and `discount_rate_new` to calculate `NPV_new`.
        *   Compute `Delta_NPV = NPV_new - NPV_orig`.
        *   Apply Materiality Flagging: Add a boolean column `material` to the results, set to `True` if $|Delta\_NPV| > 50000$ (USD 50k), otherwise `False`.
    *   **Output:** A `pandas.DataFrame` containing `loan_id`, `NPV_orig`, `NPV_new`, `Delta_NPV`, and `material` for each loan.
*   **2.3 Sensitivity Analysis Helper (Optional)**
    *   **Function:** `run_sensitivity(df_master, rate_shift_bp)`
    *   **Description:** An optional helper function that allows recomputing NPVs by shifting the discount rates (e.g., $\pm 100$ basis points). This function would internally call `calculate_npv` with adjusted rates. While not plotted, its presence supports future sensitivity testing.

#### Section 3: Model Artifacts Persistence

*   **3.1 Serialize NPV Engine**
    *   **Description:** The core NPV calculation function(s) or a class encapsulating the engine will be serialized using `joblib` and saved to `npv_engine.pkl`. This allows for re-loading the model without re-running the full notebook.
*   **3.2 Save Results**
    *   **Description:** The DataFrame containing the per-loan NPV results (`loan_id`, `NPV_orig`, `NPV_new`, `Delta_NPV`, `material`) will be saved to `npv_results.parquet` for downstream analysis or validation.

### Visualizations

The notebook will generate two primary visualizations to aid in interpreting the restructuring impact. Helper functions for plotting, as indicated in the requirements, will be utilized.

*   **3.1 Cash Flow Timeline (Stacked Area Chart)**
    *   **Purpose:** To visually represent the timing and magnitude of cash flows for both the original and restructured loan terms. This highlights shifts in payment burden and total cash flow over time.
    *   **Data:** Utilizes the `loan_cf_master` DataFrame.
    *   **Features:**
        *   **X-axis:** Payment date.
        *   **Y-axis:** Cash flow amount (sum of interest and principal for each period).
        *   **Areas:** Two distinct stacked areas, one for "Original Cash Flows" and one for "Restructured Cash Flows," using different colors.
    *   **Helper Function:** `plot_cashflow_timeline(df_long)`

*   **3.2 $\Delta$NPV Waterfall Chart**
    *   **Purpose:** To explicitly visualize the components contributing to the economic gain or loss from restructuring, showing a clear progression from the original NPV to the new NPV.
    *   **Data:** Utilizes the aggregated results DataFrame (`npv_results.parquet`).
    *   **Features:**
        *   **Bars:** Represents `NPV_orig`, then intermediate adjustments (if applicable, though simplified here to direct transition), leading to `NPV_new`, and finally a bar showing `Delta_NPV`.
        *   **Coloring:** Differentiates between gains and losses in the waterfall segments.
    *   **Helper Function:** `plot_delta_npv_waterfall(results_df)`

## 4. Additional Notes or Instructions

### Assumptions

*   **Deterministic Cash Flows:** The model assumes that all future cash flows (principal and interest payments) are known with certainty. It does not account for defaults, prepayments, or contingent payments.
*   **Discount Rate Application:** The original EIR is used as the baseline discount rate for both original and new cash flows. Adjustments for increased credit risk are applied as an additive spread only to the new cash flows.
*   **Amortization Method:** Standard loan amortization formulas are assumed for generating cash flow schedules.
*   **Materiality Threshold:** The predefined threshold for flagging material changes in NPV is USD 50,000.

### Constraints

*   **Toy Dataset Size:** The analysis is constrained to a small, synthetic dataset of approximately 10 loans.
*   **No Real Data:** No external or real-world loan data will be used; all data will be synthetically generated within the notebook.
*   **No Deployment:** This specification focuses solely on the analytical notebook content and logic; no deployment steps or platform-specific references are to be included.
*   **No Python Code:** This document is a specification; no executable Python code snippets are to be written.

### Customization Instructions

*   **Synthetic Data Parameters:** Users can modify the parameters for synthetic data generation (e.g., number of loans, ranges for principal, rates, terms) to explore different scenarios.
*   **Materiality Threshold:** The `USD 50k` materiality threshold can be adjusted within the code to suit different analytical requirements.
*   **Credit Spread Adjustment:** The magnitude of the credit spread added for worsened ratings can be modified.
*   **Sensitivity Analysis:** The `run_sensitivity` helper function, while not part of the primary visualizations, can be leveraged to explore the impact of shifts in discount rates on the calculated NPVs. This functionality is intended for use in subsequent parts of a broader study.
```