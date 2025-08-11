
import streamlit as st
import pandas as pd
from application_pages.plots import calculate_npv, run_npv_analysis_adaptable, run_sensitivity

def run_page2():
    st.header("2. NPV Calculation Engine")
    st.subheader("NPV Calculation Function")

    with st.expander("Business Value & Technical Implementation"):
        st.markdown("""
            The `calculate_npv` function is the cornerstone of our economic impact analysis.\
            It implements the fundamental Net Present Value formula:
        """)
        st.latex(r"NPV = \sum_{t=1}^{T} \frac{CF_t}{(1+r)^t}")

    st.subheader("Run NPV Analysis")

    with st.expander("Business Value & Technical Implementation"):
        st.markdown(r"""
            The `run_npv_analysis` function orchestrates the application of our core NPV calculation logic.\
            The $\Delta NPV$ is then computed as the difference:
        """)
        st.latex(r"\Delta NPV = NPV_{new} - NPV_{orig}")

    materiality_threshold = st.number_input(
        "Materiality Threshold ($)", min_value=1000.0, value=50000.0, step=1000.0, key='materiality_threshold_page2'
    )

    if st.button("Run NPV Analysis", key='run_npv_btn_page2'):
        if 'loan_cf_master_enhanced' not in st.session_state or st.session_state.loan_cf_master_enhanced.empty:
            st.warning("Please generate loan data first on Page 1.")
        else:
            with st.spinner("Running NPV analysis..."):
                @st.cache_data
                def run_npv_analysis_cached(df_master, material_thresh):
                    return run_npv_analysis_adaptable(df_master, material_thresh)

                st.session_state.npv_results_enhanced = run_npv_analysis_cached(st.session_state.loan_cf_master_enhanced, materiality_threshold)
            st.success("NPV analysis complete!")

    if 'npv_results_enhanced' in st.session_state and not st.session_state.npv_results_enhanced.empty:
        st.subheader("NPV Analysis Results")
        st.dataframe(st.session_state.npv_results_enhanced)

        st.download_button(
            label="Download npv_results.parquet",
            data=st.session_state.npv_results_enhanced.to_parquet(index=False),
            file_name="npv_results.parquet",
            mime="application/octet-stream"
        )

    st.subheader("Sensitivity Analysis")

    with st.expander("Business Value & Technical Implementation"):
        st.markdown(r"""
            Sensitivity analysis is a crucial technique in financial modeling.\
            For each loan, it then calculates new, shifted discount rates for both the original ($r_{orig, shifted}$) and new ($r_{new, shifted}$) scenarios:
        """)
        st.latex(r"r_{orig, shifted} = r_{orig} + \text{rate\_shift\_decimal}")
        st.latex(r"r_{new, shifted} = r_{new} + \text{rate\_shift\_decimal}")

    rate_shift_bp = st.number_input(
        "Rate Shift (Basis Points)", min_value=-200, max_value=200, value=0, step=10, key='rate_shift_bp_page2'
    )

    if st.button("Run Sensitivity Analysis", key='run_sensitivity_btn_page2'):
        if 'loan_cf_master_enhanced' not in st.session_state or st.session_state.loan_cf_master_enhanced.empty:
            st.warning("Please generate loan data first on Page 1.")
        else:
            with st.spinner("Running sensitivity analysis..."):
                @st.cache_data
                def run_sensitivity_cached(df_master, rate_shift):
                    return run_sensitivity(df_master, rate_shift)
                
                st.session_state.sensitivity_results_enhanced = run_sensitivity_cached(st.session_state.loan_cf_master_enhanced, rate_shift_bp)
            st.success("Sensitivity analysis complete!")

    if 'sensitivity_results_enhanced' in st.session_state and not st.session_state.sensitivity_results_enhanced.empty:
        st.subheader("Sensitivity Analysis Results (shifted rates)")
        st.dataframe(st.session_state.sensitivity_results_enhanced)
