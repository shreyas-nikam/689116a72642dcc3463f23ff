
import streamlit as st
import pandas as pd
from application_pages.utils import generate_enhanced_loan_data_adaptable, expand_cashflows, calc_discount_rate, tidy_merge

def run_page1():
    st.header("1. Data Generation and Pre-processing")
    st.subheader("Generate Synthetic Loan Data")

    with st.expander("Business Value & Technical Implementation"):
        st.markdown("""
            Generating synthetic data allows us to create a realistic, yet controlled, environment to build and validate our analytical models.\
            The `generate_enhanced_loan_data` function creates a `pandas.DataFrame` that simulates our loan portfolio.\
        """)

    num_loans = st.number_input("Number of Loans to Generate", min_value=10, max_value=100, value=25, key='num_loans_page1')
    # In a real application, you might add more controls here for data generation, e.g., restructure_rate_pct

    if st.button("Generate & Process Loan Data", key='generate_data_btn_page1'):
        with st.spinner("Generating and processing data..."):
            @st.cache_data
            def create_enhanced_loan_dataset_cached(num_loans_param):
                data = generate_enhanced_loan_data_adaptable(num_loans_param)
                df = pd.DataFrame(data)
                df['restructure_date'] = pd.to_datetime(df['restructure_date'])
                df['issuance_date'] = pd.to_datetime(df['issuance_date'])
                return df

            @st.cache_data
            def expand_cashflows_cached(df_loans):
                return expand_cashflows(df_loans)
            
            @st.cache_data
            def calc_discount_rate_cached(df_loans):
                return calc_discount_rate(df_loans)

            @st.cache_data
            def tidy_merge_cached(cf_orig, cf_new, df_loans):
                return tidy_merge(cf_orig, cf_new, df_loans)

            st.session_state.df_loans_enhanced = create_enhanced_loan_dataset_cached(num_loans)
            st.session_state.cf_orig_enhanced, st.session_state.cf_new_enhanced = expand_cashflows_cached(st.session_state.df_loans_enhanced)
            st.session_state.df_loans_with_rates_enhanced = calc_discount_rate_cached(st.session_state.df_loans_enhanced)
            st.session_state.loan_cf_master_enhanced = tidy_merge_cached(st.session_state.cf_orig_enhanced, st.session_state.cf_new_enhanced, st.session_state.df_loans_with_rates_enhanced)

        st.success("Data generation and pre-processing complete!")

    if 'df_loans_enhanced' in st.session_state and not st.session_state.df_loans_enhanced.empty:
        st.subheader("Generated Loan Data Preview")
        st.dataframe(st.session_state.df_loans_enhanced.head())
        st.subheader("Tidied Cash Flow Master Data Preview")
        st.dataframe(st.session_state.loan_cf_master_enhanced.head())

        st.download_button(
            label="Download loan_cf_master.parquet",
            data=st.session_state.loan_cf_master_enhanced.to_parquet(index=False),
            file_name="loan_cf_master.parquet",
            mime="application/octet-stream"
        )

