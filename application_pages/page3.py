
import streamlit as st
import pandas as pd
from application_pages.plots import plot_improved_cashflow_timeline_streamlit, plot_improved_waterfall_chart_streamlit, plot_loan_level_analysis_streamlit

def run_page3():
    st.header("3. Visualizations")

    if 'loan_cf_master_enhanced' in st.session_state and not st.session_state.loan_cf_master_enhanced.empty:
        st.subheader("Cash Flow Timeline")
        with st.expander("Business Value & Technical Implementation"):
            st.markdown("""
                The Cash Flow Timeline visualization provides a clear and intuitive representation...\
                **X-axis:** Represents the `date` of the payments...\
                **Y-axis:** Shows the `Cash Flow Amount`...
            """)
        
        # Selection for individual loan cash flow
        selected_loan_id = st.selectbox(
            "Select Loan ID for Detailed Cash Flow Timeline", 
            options=['All Loans'] + list(st.session_state.loan_cf_master_enhanced['loan_id'].unique()),
            key='cf_loan_id_select_page3'
        )

        df_for_cf_plot = st.session_state.loan_cf_master_enhanced
        if selected_loan_id != 'All Loans':
            df_for_cf_plot = df_for_cf_plot[df_for_cf_plot['loan_id'] == selected_loan_id]

        fig_cf = plot_improved_cashflow_timeline_streamlit(df_for_cf_plot)
        st.plotly_chart(fig_cf, use_container_width=True) # Use plotly_chart for Plotly figures

        st.subheader("\u0394NPV Waterfall Chart")
        with st.expander("Business Value & Technical Implementation"):
            st.markdown(r"""
                The $\Delta$NPV Waterfall Chart is an exceptionally powerful visualization...\
                It calculates the aggregate `NPV_orig`, `NPV_new`, and the total `Delta_NPV`...
            """)
        
        if 'npv_results_enhanced' in st.session_state and not st.session_state.npv_results_enhanced.empty:
            fig_waterfall = plot_improved_waterfall_chart_streamlit(st.session_state.npv_results_enhanced)
            st.plotly_chart(fig_waterfall, use_container_width=True)

            st.subheader("Comprehensive Loan Portfolio Analysis")
            with st.expander("Loan-Level Analysis Details"):
                st.markdown("""
                    This visualization provides a multi-faceted view of the restructuring impact across the portfolio...
                """)
            fig_loan_level = plot_loan_level_analysis_streamlit(st.session_state.npv_results_enhanced, st.session_state.df_loans_enhanced)
            st.plotly_chart(fig_loan_level, use_container_width=True)
        else:
            st.info("Run NPV Analysis on Page 2 to generate Waterfall and Loan-Level charts.")
    else:
        st.info("Generate Loan Data on Page 1 and run NPV Analysis on Page 2 to view visualizations.")
