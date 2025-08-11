
# Streamlit Application Requirements Specification: Loan Restructuring Impact Analyzer

This document outlines the requirements for developing a Streamlit application based on the provided Jupyter Notebook content and user specifications. It details interactive components, data flow, and integration of core logic.

## 1. Application Overview

The Loan Restructuring Impact Analyzer is an interactive web application designed to evaluate the economic impact of loan restructuring events. It provides a robust, transparent, and auditable framework to quantify the financial gain or loss from restructuring using a deterministic Net Present Value (NPV) modeling framework.

**Learning Goals:**
*   Create a synthetic dataset representing approximately 10 loans, including all necessary fields for a comprehensive NPV study of loan restructuring.
*   Develop a single deterministic discounted-cash-flow engine capable of computing $NPV_{orig}$, $NPV_{new}$, and $\Delta NPV$ for each loan within the dataset.
*   Interpret the results effectively through two key visualizations: one depicting cash-flow timing differences and another illustrating the economic gain/loss via a waterfall chart.
*   Persist essential model artifacts, including the NPV engine (`npv_engine.pkl`) and the calculated NPV results (`npv_results.parquet`), to facilitate seamless subsequent validation and governance processes.

## 2. User Interface Requirements

### Layout and Navigation Structure
The application will follow a clean, intuitive layout:
*   **Sidebar (`st.sidebar`):** Dedicated to input controls (data generation parameters, materiality threshold, sensitivity analysis parameters) and navigation.
*   **Main Content Area:** Displays the analysis steps, data tables, and visualizations. Content will be organized using `st.tabs` or `st.expander` for logical separation (e.g., "Data Generation," "NPV Analysis," "Visualizations").

### Input Widgets and Controls
Users will interact with the application through the following widgets:
*   **Data Generation Controls (in Sidebar/Main Area):**
    *   Number input for `num_loans` (default 25).
    *   Slider or number input for `num_restructured` (percentage of total loans, e.g., 40-70%).
    *   Multi-select for `restructure_reason` to filter generated data.
    *   Button: `st.button("Generate Synthetic Loan Data")` to trigger `create_enhanced_loan_dataset()`.
*   **NPV Analysis Controls (in Sidebar/Main Area):**
    *   Number input for `MATERIALITY_THRESHOLD` (default 50000.0).
    *   Button: `st.button("Run NPV Analysis")` to trigger the `run_npv_analysis` pipeline.
*   **Sensitivity Analysis Controls (in Sidebar/Main Area):**
    *   Number input for `rate_shift_bp` (e.g., 50 for +50bp, -100 for -100bp).
    *   Button: `st.button("Run Sensitivity Analysis")` to trigger `run_sensitivity()`.
*   **Data Display Filters:**
    *   Dropdown to select a specific `loan_id` for individual cash flow timeline visualization.
    *   Checkbox to filter `npv_results_df` to show only "material" loans.

### Visualization Components
The application will present key insights through the following:
*   **Interactive Data Tables (`st.dataframe`):**
    *   `df_loans_enhanced`: Display generated raw loan data.
    *   `npv_results_enhanced`: Display per-loan NPV analysis results (NPV_orig, NPV_new, Delta_NPV, material flag).
    *   `sensitivity_results_df`: Display sensitivity analysis results.
*   **Cash Flow Timeline (`st.pyplot`):**
    *   Displays the output of `plot_improved_cashflow_timeline(loan_cf_master_enhanced)`.
    *   X-axis: Payment date.
    *   Y-axis: Cash flow amount (interest + principal).
    *   Two colored areas/lines: original vs. restructured.
*   **$\Delta$NPV Waterfall Chart (`st.pyplot`):**
    *   Displays the output of `plot_improved_waterfall_chart(npv_results_enhanced)`.
    *   Bars showing `Total Original NPV`, `Gains`, `Losses`, `Total New NPV`, and `Total Delta_NPV`.
*   **Comprehensive Loan-Level Analysis (`st.pyplot`):**
    *   Displays the output of `plot_loan_level_analysis(npv_results_enhanced, df_loans_enhanced)`.
    *   Includes individual loan impact bars, materiality distribution, impact by loan profile, and impact vs. loan size scatter plot.

### Interactive Elements and Feedback Mechanisms
*   **Buttons:** Trigger computations and plot generation.
*   **Spinners/Progress Bars (`st.spinner`, `st.progress`):** Provide visual feedback during data generation and analysis (e.g., "Generating data...", "Running NPV calculations...").
*   **Success/Error Messages (`st.success`, `st.error`):** Confirm completion of tasks or alert users to issues.
*   **Download Buttons (`st.download_button`):** Allow users to download processed dataframes (e.g., `loan_cf_master.parquet`, `npv_results.parquet`).

## 3. Additional Requirements

### Annotation and Tooltip Specifications
*   **Charts (`matplotlib`):** Annotations on bars for `plot_improved_waterfall_chart` and `plot_loan_level_analysis` will show precise values.
*   **General Information:** Use `st.info` or `st.caption` for additional contextual information, such as business value explanations or technical implementation details for each section, replicating the markdown explanations from the notebook.

### Save the States of the Fields Properly
*   All user inputs (e.g., `num_loans`, `MATERIALITY_THRESHOLD`, `rate_shift_bp`) will be stored in `st.session_state` to ensure values persist across reruns.
*   Key generated dataframes (`df_loans_enhanced`, `cf_orig_enhanced`, `cf_new_enhanced`, `df_loans_with_rates_enhanced`, `loan_cf_master_enhanced`, `npv_results_enhanced`, `sensitivity_results_df`) will also be stored in `st.session_state` to avoid redundant computations and maintain the application's state.
*   Functions performing significant data processing or calculations will be decorated with `@st.cache_data` to cache results and improve performance on subsequent runs with the same inputs.

## 4. Notebook Content and Code Requirements

This section outlines how the content from the Jupyter Notebook will be integrated into the Streamlit application, including code stubs and relevant markdown for display.

### 4.1. Application Introduction and Overview (`st.markdown`)
The initial markdown from the notebook will be rendered in Streamlit to provide context.

```python
# In Streamlit:
st.title("Loan Restructuring Impact Analyzer")
st.markdown("""
    This Streamlit application provides a focused environment for financial analysts and students to evaluate the economic impact of loan restructuring events. It implements a deterministic Net Present Value (NPV) modeling framework to compare the present value of cash flows under original and revised loan terms.
""")
# Use st.expander for detailed sections like Business Value and Learning Goals
with st.expander("Business Value"):
    st.markdown("""
        Loan restructuring is a critical financial operation for both lenders and borrowers...
        By accurately assessing the Net Present Value (NPV) changes, financial institutions can make informed decisions...
    """)
with st.expander("Learning Goals"):
    st.markdown("""
        Upon completion of this application, users will be able to:
        *   Create a synthetic dataset representing approximately 10 loans...
        *   Develop a single deterministic discounted-cash-flow engine...
        *   Interpret the results effectively through two key visualizations...
        *   Persist essential model artifacts...
    """)
```

### 4.2. Core Financial Concepts (`st.markdown`, `st.latex`)
The fundamental NPV formula and economic gain/loss will be displayed.

```python
# In Streamlit:
st.header("Net Present Value (NPV) Concepts")
st.markdown("""
    Net Present Value (NPV) is a core concept in finance...
    The fundamental formula for Net Present Value (NPV) is given by:
""")
st.latex(r"NPV = \sum_{t=1}^{T} \frac{CF_t}{(1+r)^t}")
st.markdown(r"""
    Where:
    *   $CF_t$ represents the cash flow (interest plus principal payments) at time $t$.
    *   $t$ is the time period in which the cash flow occurs (e.g., month, quarter, year).
    *   $r$ is the discount rate per period, reflecting the time value of money and the risk associated with the cash flows.
    *   $T$ is the total number of time periods over which cash flows are projected.
""")

st.subheader("Economic Gain/Loss ($\Delta NPV$)")
st.markdown("The economic impact of a loan restructuring is measured by the change in NPV:")
st.latex(r"\Delta NPV = NPV_{new} - NPV_{orig}")
st.markdown(r"""
    *   A positive $\Delta NPV$ indicates an economic gain for the lender...
    *   A negative $\Delta NPV$ indicates an economic loss for the lender...
""")

st.subheader("Discount Rate Determination")
st.markdown(r"""
    The choice of discount rate $r$ is critical...
    $$r_{new} = r_{orig} + \text{Credit Spread Adjustment}$$
""")
# ... (rest of the markdown for discount rate determination and cash flow generation)
```

### 4.3. Data Generation and Pre-processing

This section will be presented as a workflow to the user, with `st.expander` for explanations.

#### 4.3.1. Load Raw Data / Generate Synthetic Data
Users will configure parameters and trigger data generation.

```python
# In Streamlit:
st.header("1. Data Generation and Pre-processing")
st.subheader("Generate Synthetic Loan Data")

with st.expander("Business Value & Technical Implementation"):
    st.markdown("""
        Generating synthetic data allows us to create a realistic, yet controlled, environment to build and validate our analytical models...
        The `generate_enhanced_loan_data` function creates a `pandas.DataFrame` that simulates our loan portfolio...
    """)

# Input widgets in sidebar or main area
num_loans = st.sidebar.number_input("Number of Loans to Generate", min_value=10, max_value=100, value=25, key='num_loans')
# ... (other input widgets for data generation, e.g., restructure_rate_pct)

if st.sidebar.button("Generate & Process Loan Data", key='generate_data_btn'):
    with st.spinner("Generating and processing data..."):
        # Code from notebook (simplified for Streamlit flow):
        # Using the enhanced data generation for better scenarios
        # @st.cache_data for performance
        @st.cache_data
        def create_enhanced_loan_dataset_cached(num_loans_param):
            # Adapted from generate_enhanced_loan_data and create_enhanced_loan_dataset
            # Need to modify generate_enhanced_loan_data to accept num_loans as parameter
            # For simplicity in this spec, assume it generates the required number.
            # In actual app, update generate_enhanced_loan_data signature.
            data = generate_enhanced_loan_data_adaptable(num_loans_param)
            df = pd.DataFrame(data)
            df['restructure_date'] = pd.to_datetime(df['restructure_date'])
            # ... (rest of type conversions as in create_enhanced_loan_dataset)
            return df

        # Adapt notebook code to accept num_loans as input
        # (Assuming generate_enhanced_loan_data is modified to accept num_loans)
        def generate_enhanced_loan_data_adaptable(num_loans_param):
            # ... (original generate_enhanced_loan_data logic, replace `num_loans = 25` with `num_loans = num_loans_param`)
            # Ensure imports are within the function if not global in app.py
            import random
            from datetime import datetime, timedelta
            data = []
            num_loans = num_loans_param
            # ... (rest of original function)
            return data

        # Store in session state
        st.session_state.df_loans_enhanced = create_enhanced_loan_dataset_cached(num_loans)
        
        # Step 2: Expand Cashflows
        # @st.cache_data
        @st.cache_data
        def expand_cashflows_cached(df_loans):
            # ... (full expand_cashflows function code)
            return expand_cashflows(df_loans)
        st.session_state.cf_orig_enhanced, st.session_state.cf_new_enhanced = expand_cashflows_cached(st.session_state.df_loans_enhanced)
        
        # Step 3: Calculate Discount Rates
        # @st.cache_data
        @st.cache_data
        def calc_discount_rate_cached(df_loans):
            # ... (full calc_discount_rate function code)
            return calc_discount_rate(df_loans)
        st.session_state.df_loans_with_rates_enhanced = calc_discount_rate_cached(st.session_state.df_loans_enhanced)
        
        # Step 4: Tidy and Merge Data
        # @st.cache_data
        @st.cache_data
        def tidy_merge_cached(cf_orig, cf_new, df_loans):
            # ... (full tidy_merge function code)
            return tidy_merge(cf_orig, cf_new, df_loans)
        st.session_state.loan_cf_master_enhanced = tidy_merge_cached(st.session_state.cf_orig_enhanced, st.session_state.cf_new_enhanced, st.session_state.df_loans_with_rates_enhanced)
        
    st.success("Data generation and pre-processing complete!")

if 'df_loans_enhanced' in st.session_state and not st.session_state.df_loans_enhanced.empty:
    st.subheader("Generated Loan Data Preview")
    st.dataframe(st.session_state.df_loans_enhanced.head())
    st.subheader("Tidied Cash Flow Master Data Preview")
    st.dataframe(st.session_state.loan_cf_master_enhanced.head())

# Code Stubs:
```python
# Function definitions (to be placed in app.py or a utils.py file)
def generate_enhanced_loan_data_adaptable(num_loans_param):
    """
    Generate enhanced synthetic loan data with more realistic and diverse scenarios.
    Modified to accept num_loans_param.
    """
    import random
    import pandas as pd
    from datetime import datetime, timedelta
    
    data = []
    num_loans = num_loans_param # Use parameter
    
    # ... (rest of generate_enhanced_loan_data function from notebook)
    return data

def create_enhanced_loan_dataset():
    """Create enhanced synthetic loan dataset (calls adaptable version)"""
    data = generate_enhanced_loan_data_adaptable(st.session_state.num_loans) # Read num_loans from session_state
    df = pd.DataFrame(data)
    # ... (rest of data type conversions)
    return df

def _calculate_payment_schedule_params(annual_rate, term_mths, pay_freq, principal_amount):
    # ... (full function code from notebook)

def _calculate_amortization_schedule(loan_id, principal_amount, annual_rate, term_mths, pay_freq, start_date):
    # ... (full function code from notebook)

def expand_cashflows(df_loans):
    # ... (full function code from notebook)

def calc_discount_rate(df_loans):
    # ... (full function code from notebook)

def tidy_merge(cf_orig, cf_new, df_loans):
    # ... (full function code from notebook)

# Data Persistence (via download button)
# if 'loan_cf_master_enhanced' in st.session_state:
#     st.download_button(
#         label="Download loan_cf_master.parquet",
#         data=st.session_state.loan_cf_master_enhanced.to_parquet(index=False),
#         file_name="loan_cf_master.parquet",
#         mime="application/octet-stream"
#     )
```

### 4.4. NPV Calculation Engine

This section will house the core analytical capabilities.

```python
# In Streamlit:
st.header("2. NPV Calculation Engine")
st.subheader("NPV Calculation Function")

with st.expander("Business Value & Technical Implementation"):
    st.markdown("""
        The `calculate_npv` function is the cornerstone of our economic impact analysis...
        It implements the fundamental Net Present Value formula:
    """)
    st.latex(r"NPV = \sum_{t=1}^{T} \frac{CF_t}{(1+r)^t}")

# Code Stubs:
```python
def calculate_npv(cashflows, discount_rate):
    # ... (full function code from notebook)

# Serialization of engine (done during app setup, not interactive user action)
# import joblib
# joblib.dump(calculate_npv, 'npv_engine.pkl')
```

#### 4.4.1. Run NPV Analysis
Users will trigger the NPV calculations and view results.

```python
# In Streamlit:
st.subheader("Run NPV Analysis")

with st.expander("Business Value & Technical Implementation"):
    st.markdown(r"""
        The `run_npv_analysis` function orchestrates the application of our core NPV calculation logic...
        The $\Delta NPV$ is then computed as the difference:
    """)
    st.latex(r"\Delta NPV = NPV_{new} - NPV_{orig}")

# Input for materiality threshold
materiality_threshold = st.sidebar.number_input(
    "Materiality Threshold ($)", min_value=1000.0, value=50000.0, step=1000.0, key='materiality_threshold'
)

if st.sidebar.button("Run NPV Analysis", key='run_npv_btn'):
    if 'loan_cf_master_enhanced' not in st.session_state or st.session_state.loan_cf_master_enhanced.empty:
        st.warning("Please generate loan data first.")
    else:
        with st.spinner("Running NPV analysis..."):
            # @st.cache_data
            @st.cache_data
            def run_npv_analysis_cached(df_master, material_thresh):
                # Temporarily set MATERIALITY_THRESHOLD if it's a global constant
                # In a real app, pass it as a parameter or encapsulate in a class/closure
                global MATERIALITY_THRESHOLD_GLOBAL # Use a temporary global for demonstration
                MATERIALITY_THRESHOLD_GLOBAL = material_thresh
                # Original function expects MATERIALITY_THRESHOLD from its scope.
                # Adjust `run_npv_analysis` to accept `materiality_threshold` as argument.
                return run_npv_analysis_adaptable(df_master, material_thresh)

            # Adapt notebook code to accept materiality_threshold as input
            def run_npv_analysis_adaptable(df_master: pd.DataFrame, MATERIALITY_THRESHOLD: float) -> pd.DataFrame:
                # ... (original run_npv_analysis logic, replace internal constant with parameter)
                # Ensure calculate_npv is accessible (imported or defined globally)
                # The 'material' flag is True if the absolute Delta_NPV strictly exceeds the threshold.
                # ... (rest of original function)
                return result_df
                
            st.session_state.npv_results_enhanced = run_npv_analysis_cached(st.session_state.loan_cf_master_enhanced, materiality_threshold)
        st.success("NPV analysis complete!")

if 'npv_results_enhanced' in st.session_state and not st.session_state.npv_results_enhanced.empty:
    st.subheader("NPV Analysis Results")
    st.dataframe(st.session_state.npv_results_enhanced)
    # Download button for results
    # st.download_button(
    #     label="Download npv_results.parquet",
    #     data=st.session_state.npv_results_enhanced.to_parquet(index=False),
    #     file_name="npv_results.parquet",
    #     mime="application/octet-stream"
    # )

# Code Stubs:
```python
# Global constant (or passed as argument to run_npv_analysis_adaptable)
# MATERIALITY_THRESHOLD = 50000.0 # This should be passed from Streamlit input

def run_npv_analysis_adaptable(df_master: pd.DataFrame, MATERIALITY_THRESHOLD: float) -> pd.DataFrame:
    # ... (full run_npv_analysis function code from notebook, using MATERIALITY_THRESHOLD parameter)
    # Calls calculate_npv (defined above)
    return result_df
```

#### 4.4.2. Sensitivity Analysis Helper (Optional)
Users can test the sensitivity of NPV to rate shifts.

```python
# In Streamlit:
st.subheader("Sensitivity Analysis")

with st.expander("Business Value & Technical Implementation"):
    st.markdown(r"""
        Sensitivity analysis is a crucial technique in financial modeling...
        For each loan, it then calculates new, shifted discount rates for both the original ($r_{orig, shifted}$) and new ($r_{new, shifted}$) scenarios:
    """)
    st.latex(r"r_{orig, shifted} = r_{orig} + \text{rate\_shift\_decimal}")
    st.latex(r"r_{new, shifted} = r_{new} + \text{rate\_shift\_decimal}")

# Input for rate shift in basis points
rate_shift_bp = st.sidebar.number_input(
    "Rate Shift (Basis Points)", min_value=-200, max_value=200, value=0, step=10, key='rate_shift_bp'
)

if st.sidebar.button("Run Sensitivity Analysis", key='run_sensitivity_btn'):
    if 'loan_cf_master_enhanced' not in st.session_state or st.session_state.loan_cf_master_enhanced.empty:
        st.warning("Please generate loan data first.")
    else:
        with st.spinner("Running sensitivity analysis..."):
            # @st.cache_data
            @st.cache_data
            def run_sensitivity_cached(df_master, rate_shift):
                # ... (full run_sensitivity function code, ensures calculate_npv is accessible)
                return run_sensitivity(df_master, rate_shift)
            
            st.session_state.sensitivity_results_enhanced = run_sensitivity_cached(st.session_state.loan_cf_master_enhanced, rate_shift_bp)
        st.success("Sensitivity analysis complete!")

if 'sensitivity_results_enhanced' in st.session_state and not st.session_state.sensitivity_results_enhanced.empty:
    st.subheader("Sensitivity Analysis Results (shifted rates)")
    st.dataframe(st.session_state.sensitivity_results_enhanced)

# Code Stubs:
```python
def run_sensitivity(df_master, rate_shift_bp):
    # ... (full function code from notebook)
    # Calls calculate_npv (defined above)
    return result_df
```

### 4.5. Visualizations

This section will display the generated plots.

```python
# In Streamlit:
st.header("3. Visualizations")

if 'loan_cf_master_enhanced' in st.session_state and not st.session_state.loan_cf_master_enhanced.empty:
    st.subheader("Cash Flow Timeline")
    with st.expander("Business Value & Technical Implementation"):
        st.markdown("""
            The Cash Flow Timeline visualization provides a clear and intuitive representation...
            **X-axis:** Represents the `date` of the payments...
            **Y-axis:** Shows the `Cash Flow Amount`...
        """)
    
    # Selection for individual loan cash flow
    selected_loan_id = st.selectbox(
        "Select Loan ID for Detailed Cash Flow Timeline (Optional)", 
        options=['All Loans'] + list(st.session_state.loan_cf_master_enhanced['loan_id'].unique()),
        key='cf_loan_id_select'
    )

    df_for_cf_plot = st.session_state.loan_cf_master_enhanced
    if selected_loan_id != 'All Loans':
        df_for_cf_plot = df_for_cf_plot[df_for_cf_plot['loan_id'] == selected_loan_id]

    fig_cf = plot_improved_cashflow_timeline_streamlit(df_for_cf_plot)
    st.pyplot(fig_cf) # Streamlit displays matplotlib figures

    st.subheader("$\Delta$NPV Waterfall Chart")
    with st.expander("Business Value & Technical Implementation"):
        st.markdown(r"""
            The $\Delta$NPV Waterfall Chart is an exceptionally powerful visualization...
            It calculates the aggregate `NPV_orig`, `NPV_new`, and the total `Delta_NPV`...
        """)
    
    if 'npv_results_enhanced' in st.session_state and not st.session_state.npv_results_enhanced.empty:
        fig_waterfall = plot_improved_waterfall_chart_streamlit(st.session_state.npv_results_enhanced)
        st.pyplot(fig_waterfall)

        st.subheader("Comprehensive Loan Portfolio Analysis")
        with st.expander("Loan-Level Analysis Details"):
            st.markdown("""
                This visualization provides a multi-faceted view of the restructuring impact across the portfolio...
            """)
        fig_loan_level = plot_loan_level_analysis_streamlit(st.session_state.npv_results_enhanced, st.session_state.df_loans_enhanced)
        st.pyplot(fig_loan_level)
    else:
        st.info("Run NPV Analysis to generate Waterfall and Loan-Level charts.")
else:
    st.info("Generate Loan Data and run NPV Analysis to view visualizations.")


# Code Stubs (adjusted to return figures for Streamlit's st.pyplot):
```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def plot_improved_cashflow_timeline_streamlit(df_long):
    """
    Adjusted plot_improved_cashflow_timeline to return a matplotlib figure object.
    """
    # ... (full plot_improved_cashflow_timeline function code)
    # Replace plt.show() with return fig
    # Add plt.clf() to clear figure after return to prevent re-rendering issues
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    # ... (rest of plotting code)
    plt.tight_layout()
    return fig

def plot_improved_waterfall_chart_streamlit(results_df):
    """
    Adjusted plot_improved_waterfall_chart to return a matplotlib figure object.
    """
    # ... (full plot_improved_waterfall_chart function code)
    # Replace plt.show() with return fig
    # Add plt.clf()
    fig, ax = plt.subplots(figsize=(14, 8))
    # ... (rest of plotting code)
    plt.tight_layout()
    return fig

def plot_loan_level_analysis_streamlit(npv_results_df, df_loans):
    """
    Adjusted plot_loan_level_analysis to return a matplotlib figure object.
    """
    # ... (full plot_loan_level_analysis function code)
    # Replace plt.show() with return fig
    # Add plt.clf()
    fig = plt.figure(figsize=(16, 12))
    # ... (rest of plotting code)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# General Streamlit initialization and dependencies (at the top of app.py)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib # for loading engine if needed
# Import functions from separate utils.py or define them directly
# from utils import generate_enhanced_loan_data_adaptable, _calculate_amortization_schedule, ...
# And the plot functions:
# from plots import plot_improved_cashflow_timeline_streamlit, ...

# Session State Initialization:
# if 'df_loans_enhanced' not in st.session_state:
#     st.session_state.df_loans_enhanced = pd.DataFrame()
# if 'loan_cf_master_enhanced' not in st.session_state:
#     st.session_state.loan_cf_master_enhanced = pd.DataFrame()
# if 'npv_results_enhanced' not in st.session_state:
#     st.session_state.npv_results_enhanced = pd.DataFrame()
# if 'sensitivity_results_enhanced' not in st.session_state:
#     st.session_state.sensitivity_results_enhanced = pd.DataFrame()
```
