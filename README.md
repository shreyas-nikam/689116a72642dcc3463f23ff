# QuLab: Loan Restructuring Impact Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app)

## Project Title

**QuLab: Loan Restructuring Economic Impact Analysis with NPV Modeling**

## Description

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

## Features

This Streamlit application provides a multi-faceted approach to analyzing the economic impact of loan restructuring:

*   **Synthetic Data Generation:**
    *   Generates a customizable synthetic dataset of loans with various attributes (amount, term, rate, credit score, loan type, region).
    *   Simulates loan restructuring events with new terms and rates for a subset of loans.
    *   Expands original and restructured loan terms into detailed cash flow schedules.
    *   Calculates effective discount rates for both original and new scenarios based on credit scores.
    *   Provides preview and download options for the generated and processed data.

*   **NPV Calculation Engine:**
    *   Implements a robust Net Present Value (NPV) calculation function.
    *   Orchestrates the NPV calculation for all loans, comparing `NPV_orig` and `NPV_new` to derive `Delta_NPV`.
    *   Allows setting a materiality threshold to flag significant `Delta_NPV` changes.
    *   Provides a preview and download option for the NPV results.

*   **Sensitivity Analysis:**
    *   Enables users to perform sensitivity analysis by shifting the discount rates (in basis points) for all loans.
    *   Recalculates NPVs and Delta_NPVs under the shifted rate scenario.

*   **Interactive Visualizations:**
    *   **Cash Flow Timeline:** Visualizes the `total_payment` cash flow over time for selected loans, comparing original and new scenarios.
    *   **$\Delta$NPV Waterfall Chart:** Provides an aggregate view of the total original NPV, gains, losses, and total new NPV across the entire portfolio, culminating in the total portfolio $\Delta$NPV.
    *   **Comprehensive Loan Portfolio Analysis:** A multi-plot visualization including:
        *   Individual loan impact ($\Delta$NPV) bar chart.
        *   Materiality distribution pie chart.
        *   Total $\Delta$NPV by loan type bar chart.
        *   $\Delta$NPV vs. Original Loan Amount scatter plot.
    *   All visualizations are interactive, powered by Plotly.

*   **Data Persistence:** Utilizes Streamlit's `st.session_state` and `@st.cache_data` decorators to persist and cache generated data and analysis results across different pages and reruns, enhancing user experience and performance.

## Getting Started

Follow these instructions to set up and run the application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/qu-lab-loan-restructuring.git
    cd qu-lab-loan-restructuring
    ```
    (Replace `https://github.com/your-username/qu-lab-loan-restructuring.git` with the actual repository URL)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    Create a `requirements.txt` file in the root directory of your project with the following content:
    ```
    streamlit>=1.30.0
    pandas>=2.0.0
    numpy>=1.20.0
    plotly>=5.0.0
    pyarrow>=10.0.0 # For parquet downloads
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser.

2.  **Navigate the Application:**
    The application is structured into three main pages, accessible via the sidebar navigation:

    *   **Data Generation:**
        *   Enter the desired "Number of Loans to Generate".
        *   Click "Generate & Process Loan Data". This will create synthetic loan data, expand cash flows, calculate discount rates, and prepare a master cash flow dataset.
        *   Preview the generated data and download the `loan_cf_master.parquet` file if desired.

    *   **NPV Analysis:**
        *   Set the "Materiality Threshold ($)" to define what constitutes a "material" $\Delta$NPV.
        *   Click "Run NPV Analysis" to compute `NPV_orig`, `NPV_new`, and `Delta_NPV` for each loan, and flag material changes.
        *   Preview the NPV analysis results and download `npv_results.parquet`.
        *   Adjust the "Rate Shift (Basis Points)" and click "Run Sensitivity Analysis" to see how NPVs change under different discount rate scenarios.

    *   **Visualizations:**
        *   Explore the "Cash Flow Timeline" for individual loans or the aggregate by selecting from the dropdown.
        *   View the "$\Delta$NPV Waterfall Chart" to understand the overall portfolio impact.
        *   Analyze the "Comprehensive Loan Portfolio Analysis" for multi-faceted insights into loan-level impacts, materiality distribution, impact by loan type, and $\Delta$NPV vs. loan size.
        *   *Note:* Ensure data has been generated (Page 1) and NPV analysis run (Page 2) to populate the charts on this page.

## Project Structure

The project follows a modular structure to keep the code organized and maintainable:

```
qu-lab-loan-restructuring/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # List of Python dependencies
├── application_pages/          # Directory containing individual Streamlit pages and core logic
│   ├── __init__.py             # Makes application_pages a Python package
│   ├── page1.py                # Handles data generation and pre-processing
│   ├── page2.py                # Manages NPV and sensitivity analysis calculations
│   ├── page3.py                # Contains visualization logic and plotting functions
│   ├── plots.py                # Plotly functions for generating charts
│   └── utils.py                # Utility functions for data generation, cash flow expansion, NPV calculation, etc.
└── README.md                   # Project documentation (this file)
```

## Technology Stack

*   **Streamlit:** For building the interactive web application user interface.
*   **Pandas:** For efficient data manipulation and analysis.
*   **NumPy:** For numerical operations, especially in data generation and financial calculations.
*   **Plotly:** For creating rich, interactive, and high-quality data visualizations.
*   **Python:** The core programming language used for the entire application.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name` or `bugfix/issue-description`).
3.  Make your changes and commit them (`git commit -m 'Add new feature'`).
4.  Push to your fork (`git push origin feature/your-feature-name`).
5.  Open a Pull Request to the `main` branch of this repository.

Please ensure your code adheres to good practices, includes appropriate comments, and passes any existing tests (or includes new ones if applicable).

## License

This project is licensed under the MIT License. See the `LICENSE` file (if present, otherwise assume standard MIT) for details.

## Contact

For any questions or inquiries, please contact:

**[Your Name/Organization Name]**
*   **Email:** [your_email@example.com]
*   **GitHub:** [link to your GitHub profile or organization]