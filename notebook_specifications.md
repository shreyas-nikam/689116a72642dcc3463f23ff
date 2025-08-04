
# Loan Restructuring NPV Analyzer - Jupyter Notebook Specification

## 1. Notebook Overview

**Learning Goals:**

This notebook aims to provide a hands-on experience in applying deterministic Net Present Value (NPV) valuation to loan restructuring scenarios.  Users will learn to generate cash flow schedules, construct discount curves, calculate NPV, and perform sensitivity analysis, all within a reproducible Python environment.

**Expected Outcomes:**

Upon completion of this notebook, users will be able to:

*   Understand the theoretical underpinnings of NPV valuation in the context of loan restructuring.
*   Generate comprehensive cash flow schedules for both original and restructured loans, accounting for various loan terms.
*   Select and justify appropriate discount rates based on loan characteristics and risk considerations.
*   Calculate and interpret NPV and ΔNPV to quantify the economic impact of loan restructuring.
*   Assess the robustness of NPV results through sensitivity analysis.
*   Identify and flag material restructuring scenarios based on predefined thresholds.

## 2. Mathematical and Theoretical Foundations

This section will provide the theoretical background for the NPV calculations.

**2.1. Net Present Value (NPV) Definition:**

The Net Present Value (NPV) is the sum of the present values of incoming and outgoing cash flows over a period of time.  It's used in capital budgeting to analyze the profitability of a projected investment or project.

The general formula for calculating NPV is:

$$NPV = \sum_{t=0}^{T} \frac{CF_t}{(1 + r)^t}$$

Where:

*   $CF_t$ = Cash flow at time *t*
*   *r* = Discount rate (the rate of return that could be earned on an alternative investment)
*   *t* = Time period
*   *T* = Total number of time periods

**2.2. NPV in Loan Restructuring:**

In the context of loan restructuring, the NPV is used to determine the economic impact of changing the terms of a loan. We compare the NPV of the original loan to the NPV of the restructured loan. The difference between these two NPVs (ΔNPV) indicates the gain or loss resulting from the restructuring.

The ΔNPV is calculated as follows:

$$ \Delta NPV = NPV_{new} - NPV_{orig} $$

Where:

*   $NPV_{new}$ is the Net Present Value of the restructured loan.
*   $NPV_{orig}$ is the Net Present Value of the original loan.

**2.3. Discount Rate Selection:**

The choice of discount rate is crucial for accurate NPV calculation.  A higher discount rate reflects higher risk and reduces the present value of future cash flows.  In loan restructuring, the discount rate should reflect the risk associated with the loan. A baseline approach is to use the original Effective Interest Rate (EIR) of the loan. However, it is imperative to consider any changes in the borrower's creditworthiness. If the credit risk has increased, a credit spread should be added to the original EIR to reflect the higher risk associated with the restructured loan.

Discount rate scenarios to be explored:

*   Original EIR:  Using the original loan's EIR as the discount rate.
*   EIR + Credit Spread: Adding a credit spread to the original EIR to account for increased credit risk.  The spread should be justified based on credit rating downgrades or other relevant risk indicators.

**2.4. Cash Flow Schedule Generation:**

A critical aspect of NPV calculation is the generation of accurate cash flow schedules. These schedules outline the expected cash inflows (e.g., interest payments) and outflows (e.g., principal payments) for both the original and restructured loans. For fixed-rate loans, the cash flow schedule can be derived using standard amortization formulas.  For floating-rate loans, forward rate curves or constant-rate simplifications are employed to project future interest payments.

**2.5. Floating Rate Loan Modelling:**

When dealing with floating-rate loans, the future interest rates are uncertain. Two approaches can be used to model the cash flows:

*   **Forward Rate Curve:**  Using the current forward rate curve to project future interest rates.  This approach reflects market expectations of future rates.
*   **Constant Rate Simplification:** Assuming the current rate stays constant for the entire life of the loan. This simplifies the calculations but may not be as accurate as using forward rates.

## 3. Code Requirements

**3.1. Expected Libraries:**

*   `pandas`: For data manipulation and analysis, especially working with tabular data (DataFrames).
*   `numpy`: For numerical computations, including array operations and mathematical functions.
*   `scipy`: For scientific computing, including financial functions (e.g., present value calculations), statistical analysis, and interpolation.
*   `joblib`: For efficient serialization and deserialization of Python objects (model persistence).
*   `matplotlib`: For creating static, interactive, and animated visualizations in Python.
*   `seaborn` (optional): For enhanced data visualization based on Matplotlib.
*   `pydantic`: For data validation and settings management using Python type annotations.
*   `loguru`: For comprehensive and flexible logging.

**3.2. Input/Output Expectations:**

*   **Input:** The notebook will read data from CSV files representing loan portfolios, restructuring scenarios, and yield curves.  The specific columns and formats are detailed in the "Dataset Requirements" table in the Input Context. Configuration parameters like materiality thresholds will be read from a YAML configuration file.
*   **Output:** The notebook will generate cash flow schedules, NPV results, and visualizations.  Cash flow schedules will be stored in parquet files, NPV results in CSV files, and visualizations as image files (e.g., PNG). Trained models will be saved as joblib artifacts.  An audit log in JSON-line format will track execution details.

**3.3. Algorithms and Functions:**

*   **CashFlowScheduleGenerator:**  This class will generate cash flow schedules for both original and restructured loans.  It will take loan parameters (amount, interest rate, payment frequency, maturity date), and restructuring terms (new maturity date, new interest rate, principal forgiveness, capitalized interest) as input.  The output will be a DataFrame containing the cash flow schedule.  The algorithm will handle both fixed and floating-rate loans.
*   **DiscountCurveBuilder:** This class will construct discount curves from yield curve data. It will take yield curve data (valuation date, tenor, zero rate) as input and generate a discount factor curve.  The algorithm should be able to interpolate discount factors for any given date.
*   **NPVCalculator:** This class will calculate the NPV of the original and restructured loans. It will take the cash flow schedules and discount factors as input and calculate the NPV using the formula: $$NPV = \sum_{t=1}^{T} \frac{CF_t}{(1 + r)^t}$$.  The output will include $NPV_{orig}$, $NPV_{new}$, and $\Delta NPV$.
*   **SensitivityAnalyzer:** This class will perform sensitivity analysis on key assumptions. It will take the NPV results and a range of discount rates (e.g., ±100 bp) as input and recalculate the NPV. The output will show the impact of the discount rate change on the $\Delta NPV$.

**3.4. Visualizations:**

*   **Cash Flow Timeline Plot:**  A stacked area or line chart showing the principal and interest components of the cash flow schedule for both the original and restructured loans.  Monthly granularity is preferred.
*   **Discount Factor Curve:** A line plot of the zero/forward rates used in the NPV calculation.
*   **NPV Bar/Waterfall Chart:** A bar or waterfall chart comparing $NPV_{orig}$ and $NPV_{new}$ for each loan, with the $\Delta NPV$ annotated.
*   **Sensitivity Tornado Chart:**  A tornado chart showing the impact of changes in key assumptions (e.g., discount rate) on the $\Delta NPV$ for a selected loan.
*   **Portfolio Heatmap/Scatter Plot:** A heatmap or scatter plot visualizing the $\Delta NPV$ against borrower rating to identify concentrations of economic losses.

All plots should be reproducible using Matplotlib and saved as image files.

## 4. Additional Notes or Instructions

*   **Data Assumptions:** The notebook assumes that the input data is clean and properly formatted.  Data validation using `pydantic` is recommended.
*   **Reproducibility:**  A fixed `random_state=42` should be used wherever randomness is involved (e.g., in generating synthetic data).
*   **Materiality Thresholds:**  The notebook should use a configuration file (`npv_config.yml`) to define materiality thresholds for flagging significant $\Delta NPV$ values. Example parameters: `material_delta_npv_abs = 100000`, `material_delta_npv_pct_portfolio = 0.01`.
*   **Audit Logging:**  The notebook should write an audit log (`audit_log_part1.jsonl`) capturing key information about each run, including the timestamp, user, run ID, threshold flags, and locations of saved artifacts.
*   **Modular Design:**  The code should be organized into modular classes (`CashFlowScheduleGenerator`, `DiscountCurveBuilder`, `NPVCalculator`, `SensitivityAnalyzer`) with clear docstrings and unit tests.
*   **Model Persistence:** Save the trained model artifacts to the `models_part1/` directory with the specified filenames (`cashflow_generator_taiwan.joblib`, `discount_curve_builder_taiwan.joblib`, `npv_calculator_taiwan.joblib`, `sensitivity_analyzer_taiwan.joblib`).
*   **Documentation Hook:**  The notebook should include functionality to auto-populate a Markdown "Case Summary" template with relevant information about each loan restructuring scenario. This should include original and new loan terms, discount rate rationale, NPV figures, and qualitative notes.
*   **Version Control:** The notebook and artifacts should be committed under the tag `npv_part1_v1.0`.

