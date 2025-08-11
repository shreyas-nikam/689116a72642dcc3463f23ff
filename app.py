
import streamlit as st
import pandas as pd

# Initialize session state for data persistence across pages
if 'df_loans_enhanced' not in st.session_state:
    st.session_state.df_loans_enhanced = pd.DataFrame()
if 'cf_orig_enhanced' not in st.session_state:
    st.session_state.cf_orig_enhanced = pd.DataFrame()
if 'cf_new_enhanced' not in st.session_state:
    st.session_state.cf_new_enhanced = pd.DataFrame()
if 'df_loans_with_rates_enhanced' not in st.session_state:
    st.session_state.df_loans_with_rates_enhanced = pd.DataFrame()
if 'loan_cf_master_enhanced' not in st.session_state:
    st.session_state.loan_cf_master_enhanced = pd.DataFrame()
if 'npv_results_enhanced' not in st.session_state:
    st.session_state.npv_results_enhanced = pd.DataFrame()
if 'sensitivity_results_enhanced' not in st.session_state:
    st.session_state.sensitivity_results_enhanced = pd.DataFrame()

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we delve into the critical financial operation of loan restructuring and its economic impact using Net Present Value (NPV) models. This interactive Streamlit application provides a robust, transparent, and auditable framework to quantify the financial gain or loss from restructuring through a deterministic NPV modeling framework.

### Business Value
Loan restructuring is a crucial tool for financial institutions to manage distressed assets, mitigate risks, and optimize portfolio performance. By accurately assessing the Net Present Value (NPV) changes resulting from restructuring, financial institutions can make informed decisions that impact their profitability and balance sheet health. This analysis helps in:
- **Risk Management:** Quantifying potential losses or gains from altered loan terms.
- **Strategic Decision Making:** Guiding decisions on whether to restructure a loan and under what terms.
- **Regulatory Compliance:** Providing an auditable framework for valuing restructured assets.
- **Portfolio Optimization:** Identifying loans where restructuring can yield the most significant positive impact.

### Learning Goals
Upon interacting with this application, users will be able to:
- **Create a Synthetic Dataset:** Generate a dataset representing various loan scenarios, essential for comprehensive NPV studies.
- **Develop a Discounted Cash Flow Engine:** Understand and apply a deterministic discounted-cash-flow engine to compute $NPV_{orig}$, $NPV_{new}$, and $\Delta NPV$ for individual loans.
- **Interpret Results Visually:** Analyze cash-flow timing differences and economic gain/loss through interactive visualizations like cash flow timelines and waterfall charts.
- **Perform Sensitivity Analysis:** Evaluate the impact of changes in discount rates on the NPV outcomes.
- **Persist Model Artifacts:** Understand the importance of saving model outputs for subsequent validation and governance.

### Core Financial Concepts
**Net Present Value (NPV):** NPV is a fundamental financial metric used to evaluate the profitability of an investment or project. It measures the difference between the present value of cash inflows and the present value of cash outflows over a period of time. The fundamental formula for Net Present Value (NPV) is given by:
$$
NPV = \sum_{t=1}^{T} \frac{CF_t}{(1+r)^t}
$$
Where:
- $CF_t$ represents the cash flow (interest plus principal payments) at time $t$.
- $t$ is the time period in which the cash flow occurs (e.g., month, quarter, year).
- $r$ is the discount rate per period, reflecting the time value of money and the risk associated with the cash flows.
- $T$ is the total number of time periods over which cash flows are projected.

**Economic Gain/Loss ($\Delta NPV$):** The economic impact of a loan restructuring is measured by the change in NPV:
$$
\Delta NPV = NPV_{new} - NPV_{orig}
$$
- A **positive $\Delta NPV$** indicates an economic gain for the lender, meaning the present value of the new cash flows is higher than the original.
- A **negative $\Delta NPV$** indicates an economic loss for the lender, meaning the present value of the new cash flows is lower than the original.

**Discount Rate Determination:** The choice of discount rate $r$ is critical as it directly impacts the calculated NPV. It typically reflects the cost of capital, risk-free rate, and a credit spread commensurate with the borrower's credit risk. For restructured loans, the new discount rate might also reflect changes in perceived risk or market conditions:
$$
r_{new} = r_{orig} + \text{Credit Spread Adjustment}
$$
""")

# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Data Generation", "NPV Analysis", "Visualizations"])

if page == "Data Generation":
    from application_pages.page1 import run_page1
    run_page1()
elif page == "NPV Analysis":
    from application_pages.page2 import run_page2
    run_page2()
elif page == "Visualizations":
    from application_pages.page3 import run_page3
    run_page3()
# Your code ends


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@quantuniversity.com](mailto:info@quantuniversity.com)
''')
