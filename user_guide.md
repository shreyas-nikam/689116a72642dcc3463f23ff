id: 689116a72642dcc3463f23ff_user_guide
summary: Lab 6.1 Net Present Value (NPV) Models - Development  User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Understanding the Economic Impact of Loan Restructuring: An NPV Analysis

## Introduction: Understanding Loan Restructuring and NPV
Duration: 0:05:00

In this codelab, we will explore the critical financial operation of loan restructuring and its economic impact using Net Present Value (NPV) models. This interactive application provides a robust, transparent, and auditable framework to quantify the financial gain or loss from restructuring through a deterministic NPV modeling framework.

### Business Value

Loan restructuring is a crucial tool for financial institutions to manage distressed assets, mitigate risks, and optimize portfolio performance. By accurately assessing the Net Present Value (NPV) changes resulting from restructuring, financial institutions can make informed decisions that impact their profitability and balance sheet health. This analysis helps in:
*   **Risk Management:** Quantifying potential losses or gains from altered loan terms.
*   **Strategic Decision Making:** Guiding decisions on whether to restructure a loan and under what terms.
*   **Regulatory Compliance:** Providing an auditable framework for valuing restructured assets.
*   **Portfolio Optimization:** Identifying loans where restructuring can yield the most significant positive impact.

### Learning Goals

Upon interacting with this application, you will be able to:
*   **Create a Synthetic Dataset:** Generate a dataset representing various loan scenarios, essential for comprehensive NPV studies.
*   **Develop a Discounted Cash Flow Engine:** Understand and apply a deterministic discounted-cash-flow engine to compute $NPV_{orig}$, $NPV_{new}$, and $\Delta NPV$ for individual loans.
*   **Interpret Results Visually:** Analyze cash-flow timing differences and economic gain/loss through interactive visualizations like cash flow timelines and waterfall charts.
*   **Perform Sensitivity Analysis:** Evaluate the impact of changes in discount rates on the NPV outcomes.
*   **Persist Model Artifacts:** Understand the importance of saving model outputs for subsequent validation and governance.

### Core Financial Concepts

**Net Present Value (NPV):** NPV is a fundamental financial metric used to evaluate the profitability of an investment or project. It measures the difference between the present value of cash inflows and the present value of cash outflows over a period of time. The fundamental formula for Net Present Value (NPV) is given by:

$$
NPV = \sum_{t=1}^{T} \frac{CF_t}{(1+r)^t}
$$

Where:
*   $CF_t$ represents the cash flow (interest plus principal payments) at time $t$.
*   $t$ is the time period in which the cash flow occurs (e.g., month, quarter, year).
*   $r$ is the discount rate per period, reflecting the time value of money and the risk associated with the cash flows.
*   $T$ is the total number of time periods over which cash flows are projected.

**Economic Gain/Loss ($\Delta NPV$):** The economic impact of a loan restructuring is measured by the change in NPV:

$$
\Delta NPV = NPV_{new} - NPV_{orig}
$$

*   A **positive $\Delta NPV$** indicates an economic gain for the lender, meaning the present value of the new cash flows is higher than the original.
*   A **negative $\Delta NPV$** indicates an economic loss for the lender, meaning the present value of the new cash flows is lower than the original.

**Discount Rate Determination:** The choice of discount rate $r$ is critical as it directly impacts the calculated NPV. It typically reflects the cost of capital, risk-free rate, and a credit spread commensurate with the borrower's credit risk. For restructured loans, the new discount rate might also reflect changes in perceived risk or market conditions:

$$
r_{new} = r_{orig} + \text{Credit Spread Adjustment}
$$

<aside class="positive">
<b>Key Takeaway:</b> This application helps financial professionals understand the precise financial implications of changing loan terms, enabling data-driven decisions that optimize portfolio performance and manage risk effectively.
</aside>

## Step 1: Generating and Preparing Loan Data
Duration: 0:03:00

In this step, we will generate a synthetic dataset of loans and process them to create detailed cash flow schedules, which are essential inputs for our NPV analysis.

### Why Synthetic Data?

In financial modeling, especially when dealing with sensitive information like loan portfolios, using synthetic data is a powerful approach. It allows us to:
*   **Simulate Realistic Scenarios:** Create diverse loan characteristics and restructuring patterns without using real, confidential data.
*   **Control Variables:** Manipulate parameters (like the number of loans or restructuring rates) to test different hypotheses and understand their impact.
*   **Develop and Test Models:** Build and validate analytical models in a controlled environment before deploying them with real data.

### How to Generate and Process Data

Navigate to the "Data Generation" page using the sidebar.

1.  **Set the Number of Loans:** You will see an input field labeled "Number of Loans to Generate". This allows you to specify how many synthetic loan records the application should create. For this codelab, we recommend starting with the default value (e.g., 25) or choosing a number between 10 and 100.

    <pre><code class="console">Number of Loans to Generate [25]</code></pre>

2.  **Generate Data:** Click the "Generate & Process Loan Data" button. The application will simulate a loan portfolio, including some loans that are "restructured" with new terms (e.g., extended maturity, adjusted interest rates). It then generates detailed payment schedules for both the *original* terms of all loans and the *new* (or identical, if not restructured) terms. Finally, it assigns a specific discount rate to each loan based on its characteristics (like credit score).

    <pre><code class="console">Generate & Process Loan Data</code></pre>

3.  **Review Data Previews:** Once the processing is complete, you will see two data previews:
    *   **Generated Loan Data Preview:** This shows the initial characteristics of the synthetic loans, such as original amount, term, rate, credit score, and whether they were restructured.
    *   **Tidied Cash Flow Master Data Preview:** This is a crucial output. It combines the detailed payment schedules for both the original and new scenarios for all loans. Each row represents a single payment, specifying the loan ID, payment number, date, principal/interest components, and the remaining balance. This structured data is ready for NPV calculation.

4.  **Download Data (Optional):** You can download the `loan_cf_master.parquet` file. This file contains the complete, tidied cash flow data, which is useful for audit, further analysis, or loading into other systems.

<aside class="positive">
<b>Concept Highlight:</b> The cash flow generation simulates how loans are amortized. An amortization schedule breaks down each payment into principal and interest components over the life of the loan. This is critical because NPV relies on a stream of future cash flows.
</aside>

## Step 2: Calculating Net Present Value (NPV)
Duration: 0:04:00

In this step, we will apply the Net Present Value (NPV) calculation to quantify the economic impact of loan restructuring. This is where we determine if a restructuring action is financially beneficial or detrimental.

### The NPV Calculation Engine

Navigate to the "NPV Analysis" page using the sidebar.

The application uses the fundamental NPV formula ($NPV = \sum_{t=1}^{T} \frac{CF_t}{(1+r)^t}$) to evaluate the present value of all future cash flows for each loan. It performs this calculation for two scenarios:
1.  **Original Scenario ($NPV_{orig}$):** The NPV of cash flows assuming the loan continues under its initial terms.
2.  **New Scenario ($NPV_{new}$):** The NPV of cash flows reflecting the restructured terms (or the original terms if the loan was not restructured).

The core insight comes from comparing these two values:
*   **Economic Gain/Loss ($\Delta NPV$):** Calculated as $\Delta NPV = NPV_{new} - NPV_{orig}$. A positive $\Delta NPV$ indicates that the restructuring improved the present value for the lender (an economic gain), while a negative value indicates a loss.

### Running the NPV Analysis

1.  **Set Materiality Threshold:** Before running the analysis, you can set a "Materiality Threshold". This value helps in identifying changes in NPV that are considered significant from a business perspective. For instance, if you set it to \$50,000, any $\Delta NPV$ (positive or negative) with an absolute value greater than \$50,000 will be flagged as "material."

    <pre><code class="console">Materiality Threshold ($) [50000.0]</code></pre>

2.  **Execute Analysis:** Click the "Run NPV Analysis" button. The application will iterate through each loan, apply the NPV formula for both scenarios using the appropriate discount rates (calculated in Step 1), and compute the $\Delta NPV$.

    <pre><code class="console">Run NPV Analysis</code></pre>

3.  **Review Results:** The "NPV Analysis Results" table will display a summary for each loan, including:
    *   `loan_id`: Unique identifier for the loan.
    *   `NPV_orig`: The Net Present Value of the loan under its original terms.
    *   `NPV_new`: The Net Present Value of the loan under its new (restructured) terms.
    *   `Delta_NPV`: The difference between $NPV_{new}$ and $NPV_{orig}$, representing the economic gain or loss from restructuring.
    *   `material`: A boolean flag indicating whether the `Delta_NPV` exceeds the set materiality threshold.

4.  **Download Results (Optional):** You can download the `npv_results.parquet` file for detailed record-keeping and further analysis.

<aside class="positive">
<b>Best Practice:</b> The materiality threshold is crucial for focusing attention on the most impactful restructurings, allowing businesses to prioritize their efforts effectively.
</aside>

## Step 3: Performing Sensitivity Analysis
Duration: 0:03:00

Sensitivity analysis is a vital part of financial modeling. It helps us understand how robust our NPV results are to changes in key assumptions, particularly the discount rate, which can fluctuate due to market conditions or changes in perceived risk.

### Understanding Sensitivity Analysis

Still on the "NPV Analysis" page, scroll down to the "Sensitivity Analysis" section.

The `run_sensitivity` function in the application allows you to simulate changes in the discount rates and observe their impact on the calculated NPVs. This is achieved by shifting the discount rates for both the original and new scenarios by a specified number of basis points (bp).

*   **Basis Points (bp):** A basis point is a common unit of measure in finance, equal to one-hundredth of one percent (0.01%). So, 100 basis points equals 1 percentage point.
*   The shifted rates are calculated as:
    $$
    r_{orig, shifted} = r_{orig} + \text{rate\_shift\_decimal}
    $$
    $$
    r_{new, shifted} = r_{new} + \text{rate\_shift\_decimal}
    $$
    where `rate_shift_decimal` is the basis point shift converted to a decimal (e.g., 50 bp = 0.0050).

### How to Run Sensitivity Analysis

1.  **Input Rate Shift (Basis Points):** Use the input field "Rate Shift (Basis Points)" to specify how much you want to adjust the discount rates.
    *   A positive value (e.g., 50) will increase the discount rates by that many basis points.
    *   A negative value (e.g., -50) will decrease the discount rates.

    <pre><code class="console">Rate Shift (Basis Points) [0]</code></pre>

2.  **Execute Sensitivity:** Click the "Run Sensitivity Analysis" button. The application will recalculate the NPVs for all loans using these shifted discount rates, providing a new perspective on the economic impact under different market assumptions.

    <pre><code class="console">Run Sensitivity Analysis</code></pre>

3.  **Review Sensitivity Results:** The "Sensitivity Analysis Results (shifted rates)" table will show:
    *   The shifted original and new discount rates.
    *   The recalculated NPVs for both original and new scenarios under these shifted rates.
    *   The `Delta_NPV_Shifted`, which is the difference between the new shifted NPV and the original shifted NPV.

<aside class="negative">
<b>Important Note:</b> While this model focuses on discount rate sensitivity, real-world sensitivity analysis often considers other variables like prepayment rates, default rates, and recovery rates. This application provides a foundational understanding.
</aside>

## Step 4: Visualizing Economic Impact
Duration: 0:05:00

Visualizations are incredibly powerful for understanding complex financial data. In this step, we will explore various plots that graphically represent the cash flow dynamics and the economic impact of loan restructuring across the portfolio.

### Navigating to Visualizations

Navigate to the "Visualizations" page using the sidebar.

<aside class="positive">
<b>Prerequisite:</b> Ensure you have completed Step 1 (Data Generation) and Step 2 (NPV Analysis) before proceeding, as these visualizations rely on the generated and analyzed data. If not, the charts will display "No data" messages.
</aside>

### Cash Flow Timeline

This plot provides a clear and intuitive representation of the payment streams over time for selected loans.
*   **X-axis:** Represents the `date` of the payments.
*   **Y-axis:** Shows the `Total Payment Amount` (principal + interest).
*   **Lines:** One line for the 'Original' cash flow scenario (blue) and another for the 'New' (restructured) cash flow scenario (red).

1.  **Select Loan ID:** Use the dropdown "Select Loan ID for Detailed Cash Flow Timeline" to choose a specific loan to view its cash flow timeline. You can also select "All Loans" to see an aggregated view (though it might be less clear due to overlapping lines).

    <pre><code class="console">Select Loan ID for Detailed Cash Flow Timeline [All Loans]</code></pre>

2.  **Observe Differences:** Pay attention to how the payment amounts and the duration of payments change between the original and new scenarios, especially for restructured loans. You'll often see the 'New' line extend further out in time with potentially lower periodic payments if the term was extended.

### $\Delta$NPV Waterfall Chart

The $\Delta$NPV Waterfall Chart is an exceptionally powerful visualization for understanding the aggregate economic impact at a portfolio level. It visually breaks down the total change in NPV.
*   It starts with the `Total Original NPV` of the portfolio.
*   It then shows the sum of `Gains` (positive $\Delta NPV$ from individual loans) and `Losses` (negative $\Delta NPV$ from individual loans) as intermediate steps.
*   It concludes with the `Total New NPV` of the portfolio and the `Total $\Delta$NPV`.

This chart immediately highlights the overall positive or negative financial impact of the restructuring actions across your entire loan portfolio.

### Comprehensive Loan Portfolio Analysis

This section provides a multi-faceted view of the restructuring impact across the portfolio, using four linked subplots:

1.  **Individual Loan Impact ($\Delta$NPV):** A bar chart showing the $\Delta$NPV for each individual loan. This helps you quickly identify which specific loans contributed the most to gains or losses. Green bars indicate gains, red bars indicate losses.

2.  **Materiality Distribution:** A pie chart showing the proportion of loans that were flagged as "material" (i.e., their $\Delta$NPV exceeded the materiality threshold set in Step 2) versus "non-material." This helps in understanding the concentration of significant impacts.

3.  **Total $\Delta$NPV by Loan Type:** A bar chart aggregating the $\Delta$NPV by different loan types (e.g., Mortgage, Auto, Personal, Business). This can reveal which types of loans are most affected by restructuring or offer the largest opportunities for economic gain.

4.  **$\Delta$NPV vs. Original Loan Amount:** A scatter plot visualizing the relationship between the original size of a loan and its corresponding $\Delta$NPV. This helps in understanding if larger loans tend to have a disproportionately higher or lower impact, or if there's any correlation between loan size and restructuring outcomes. The points are color-coded by their $\Delta$NPV value.

<aside class="positive">
<b>Application:</b> These visualizations are invaluable for presenting the results of your NPV analysis to stakeholders, offering clear, actionable insights into the financial health and strategic implications of your loan restructuring efforts.
</aside>
