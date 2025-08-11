import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_improved_cashflow_timeline_streamlit(df_long):
    """
    Plots the cash flow timeline for original and new scenarios using Plotly.
    df_long is expected to have 'loan_id', 'date', 'total_payment', and 'scenario' columns.
    """
    if df_long.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Cash Flow Timeline (No data)",
            xaxis_title="Payment Date",
            yaxis_title="Total Payment Amount",
            height=500,
            template="plotly_white"
        )
        return fig

    # Filter for the relevant scenarios (Original and New)
    df_plot = df_long[df_long['scenario'].isin(['Original', 'New'])]

    fig = go.Figure()

    # Plot original cash flows
    df_orig = df_plot[df_plot['scenario'] == 'Original']
    if not df_orig.empty:
        fig.add_trace(go.Scatter(
            x=df_orig['date'],
            y=df_orig['total_payment'],
            mode='lines',
            name='Original Cash Flow',
            line=dict(color='blue', width=2)
        ))

    # Plot new cash flows
    df_new = df_plot[df_plot['scenario'] == 'New']
    if not df_new.empty:
        fig.add_trace(go.Scatter(
            x=df_new['date'],
            y=df_new['total_payment'],
            mode='lines',
            name='New Cash Flow',
            line=dict(color='red', width=2)
        ))

    fig.update_layout(
        title=f"Cash Flow Timeline for Loan ID: {df_long['loan_id'].iloc[0]}" if df_long['loan_id'].nunique() == 1 else "Cash Flow Timeline",
        xaxis_title="Payment Date",
        yaxis_title="Total Payment Amount",
        hovermode="x unified",
        height=500,
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1)
    )

    return fig

def plot_improved_waterfall_chart_streamlit(results_df):
    """
    Plots a waterfall chart of NPV components using Plotly.
    results_df is expected to have 'NPV_orig', 'NPV_new', 'Delta_NPV' columns.
    """
    if results_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="\u0394NPV Waterfall Chart (No data)",
            xaxis_title="",
            yaxis_title="NPV Amount ($)",
            height=500,
            template="plotly_white"
        )
        return fig

    # Aggregate values
    total_original_npv = results_df['NPV_orig'].sum()
    total_new_npv = results_df['NPV_new'].sum()
    total_delta_npv = results_df['Delta_NPV'].sum()

    # Calculate gains and losses based on individual loan delta NPVs
    gains = results_df[results_df['Delta_NPV'] > 0]['Delta_NPV'].sum()
    losses = results_df[results_df['Delta_NPV'] < 0]['Delta_NPV'].sum()

    # Data for waterfall chart
    # Starting point, intermediate changes, and ending point
    data = dict(
        measure=['absolute', 'relative', 'relative', 'absolute', 'total'],
        x=['Total Original NPV', 'Gains', 'Losses', 'Total New NPV', 'Total \u0394NPV'],
        y=[total_original_npv, gains, losses, total_new_npv, total_delta_npv],
        text=[f'${total_original_npv:,.2f}', f'${gains:,.2f}', f'${losses:,.2f}', f'${total_new_npv:,.2f}', f'${total_delta_npv:,.2f}'],
        textposition='outside',
        connector={ "line": { "color": "rgb(63, 63, 63)" } },
    )

    fig = go.Figure(go.Waterfall(data,
                                 increasing=dict(marker_color="teal"),
                                 decreasing=dict(marker_color="salmon"),
                                 totals=dict(marker_color="dimgray"))
                   )

    fig.update_layout(
        title="\u0394NPV Waterfall Chart (Aggregate Portfolio Impact)",
        xaxis_title="",
        yaxis_title="NPV Amount ($)",
        height=600,
        template="plotly_white",
        showlegend=False,
    )
    return fig

def plot_loan_level_analysis_streamlit(npv_results_df, df_loans):
    """
    Plots a comprehensive multi-plot visualization for loan-level analysis using Plotly.
    Combines bar chart for individual loan impact, pie chart for materiality distribution,
    bar chart for impact by loan type, and scatter plot for impact vs. loan size.
    """
    if npv_results_df.empty or df_loans.empty:
        fig = go.Figure()
        fig.update_layout(title="Loan-Level Analysis (No data)", height=700, template="plotly_white")
        return fig

    # Merge to get loan type for analysis
    merged_df = npv_results_df.merge(df_loans[['loan_id', 'loan_type', 'original_amount']], on='loan_id', how='left')
    merged_df['material_flag_str'] = merged_df['material'].apply(lambda x: 'Material' if x else 'Non-Material')

    fig = go.Figure()

    # Subplot 1: Individual Loan Impact (Bar Chart)
    fig.add_trace(go.Bar(
        x=merged_df['loan_id'],
        y=merged_df['Delta_NPV'],
        marker_color=merged_df['Delta_NPV'].apply(lambda x: 'teal' if x >= 0 else 'salmon'),
        name='Individual \u0394NPV',
        showlegend=False
    ), row=1, col=1)
    fig.update_layout(xaxis_title_text="Loan ID", yaxis_title_text="\u0394NPV ($)",
                      title_text="Individual Loan Impact (\u0394NPV)", title_y=0.98, title_x=0.5)

    # Subplot 2: Materiality Distribution (Pie Chart)
    materiality_counts = merged_df['material_flag_str'].value_counts()
    fig.add_trace(go.Pie(
        labels=materiality_counts.index,
        values=materiality_counts.values,
        name='Materiality Distribution',
        pull=[0.05 if label == 'Material' else 0 for label in materiality_counts.index],
        marker_colors=['darkorange', 'lightgray'],
        showlegend=True, legendgroup='materiality', legendgrouptitle_text='Materiality',
        textinfo='percent+label',
        domain={'x': [0.5, 1.0], 'y': [0.5, 1.0]}
    ), row=1, col=2)
    fig.update_layout(title_text="Materiality Distribution", title_x=0.75, title_y=0.98)

    # Subplot 3: Impact by Loan Type (Bar Chart)
    impact_by_loan_type = merged_df.groupby('loan_type')['Delta_NPV'].sum().reset_index()
    fig.add_trace(go.Bar(
        x=impact_by_loan_type['loan_type'],
        y=impact_by_loan_type['Delta_NPV'],
        marker_color=impact_by_loan_type['Delta_NPV'].apply(lambda x: 'darkblue' if x >= 0 else 'darkred'),
        name='Impact by Loan Type',
        showlegend=False
    ), row=2, col=1)
    fig.update_layout(xaxis_title_text="Loan Type", yaxis_title_text="Total \u0394NPV ($)",
                      title_text="Total \u0394NPV by Loan Type", title_x=0.25, title_y=0.48)

    # Subplot 4: \u0394NPV vs. Loan Size (Scatter Plot)
    fig.add_trace(go.Scatter(
        x=merged_df['original_amount'],
        y=merged_df['Delta_NPV'],
        mode='markers',
        marker=dict(
            size=10,
            color=merged_df['Delta_NPV'], # Color points by Delta_NPV value
            colorscale='Viridis', # Choose a colorscale
            colorbar_title='\u0394NPV',
            showscale=True
        ),
        name='\u0394NPV vs. Loan Size',
        showlegend=False
    ), row=2, col=2)
    fig.update_layout(xaxis_title_text="Original Loan Amount ($)", yaxis_title_text="\u0394NPV ($)",
                      title_text="\u0394NPV vs. Original Loan Amount", title_x=0.75, title_y=0.48)

    # Create subplots layout
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Individual Loan Impact (\u0394NPV)', 
                                          'Materiality Distribution', 
                                          'Total \u0394NPV by Loan Type', 
                                          '\u0394NPV vs. Original Loan Amount'),
                        specs=[[{}, {'type':'domain'}], [{}, {}]])


    # Add traces to the subplots
    # Individual Loan Impact
    fig.add_trace(go.Bar(
        x=merged_df['loan_id'],
        y=merged_df['Delta_NPV'],
        marker_color=merged_df['Delta_NPV'].apply(lambda x: 'teal' if x >= 0 else 'salmon'),
        name='Individual \u0394NPV',
        showlegend=False
    ), row=1, col=1)

    # Materiality Distribution
    fig.add_trace(go.Pie(
        labels=materiality_counts.index,
        values=materiality_counts.values,
        name='Materiality Distribution',
        pull=[0.05 if label == 'Material' else 0 for label in materiality_counts.index],
        marker_colors=['darkorange', 'lightgray'],
        showlegend=True, legendgroup='materiality', legendgrouptitle_text='Materiality',
        textinfo='percent+label',
    ), row=1, col=2)

    # Impact by Loan Type
    fig.add_trace(go.Bar(
        x=impact_by_loan_type['loan_type'],
        y=impact_by_loan_type['Delta_NPV'],
        marker_color=impact_by_loan_type['Delta_NPV'].apply(lambda x: 'darkblue' if x >= 0 else 'darkred'),
        name='Impact by Loan Type',
        showlegend=False
    ), row=2, col=1)

    # \u0394NPV vs. Loan Size
    fig.add_trace(go.Scatter(
        x=merged_df['original_amount'],
        y=merged_df['Delta_NPV'],
        mode='markers',
        marker=dict(
            size=10,
            color=merged_df['Delta_NPV'], 
            colorscale='Viridis', 
            colorbar_title='\u0394NPV',
            showscale=True
        ),
        name='\u0394NPV vs. Loan Size',
        showlegend=False
    ), row=2, col=2)

    fig.update_layout(
        height=700,
        showlegend=True,
        template="plotly_white",
        title_text="Comprehensive Loan Portfolio Analysis"
    )

    return fig

from plotly.subplots import make_subplots


def calculate_npv(cashflows, discount_rate):
    """
    Calculates the Net Present Value (NPV) given a series of cash flows and a discount rate.
    Assumes cashflows are a Series where index is time period (e.g., 1, 2, 3...) and values are cash amounts.
    """
    if cashflows.empty:
        return 0.0
    
    # Ensure cashflows are numeric and index is suitable for period calculation
    cashflows = pd.to_numeric(cashflows)
    
    # Calculate present value for each cash flow
    # Assuming cashflows index represents periods starting from 1
    # If cashflows were a DataFrame with a 'date' column, we'd calculate time_diff_years
    # For this simplified NPV, assume `cashflows` is a series where index is `t`
    pv_sum = 0
    for t, cf_t in cashflows.items(): # Iterate through index (t) and value (cf_t)
        pv_sum += cf_t / ((1 + discount_rate)**t)
    return pv_sum

def run_npv_analysis_adaptable(df_master: pd.DataFrame, MATERIALITY_THRESHOLD: float) -> pd.DataFrame:
    """
    Orchestrates the NPV calculation for all loans in the master cash flow dataframe.
    Computes NPV_orig, NPV_new, Delta_NPV, and a materiality flag.
    """
    if df_master.empty:
        return pd.DataFrame()

    # Group by loan_id and scenario, then apply NPV calculation
    npv_results = []

    for loan_id in df_master['loan_id'].unique():
        loan_data = df_master[df_master['loan_id'] == loan_id].copy()

        # Get the single original and new discount rates for the loan
        # Assuming original_discount_rate and new_discount_rate are consistent per loan_id
        orig_rate = loan_data['original_discount_rate'].iloc[0]
        new_rate = loan_data['new_discount_rate'].iloc[0]

        # Prepare cash flows for NPV calculation
        # The `calculate_npv` function expects a Series indexed by time period.
        # We need to ensure the `payment_number` is the time index.
        
        cf_orig = loan_data[loan_data['scenario'] == 'Original'].set_index('payment_number')['total_payment']
        cf_new = loan_data[loan_data['scenario'] == 'New'].set_index('payment_number')['total_payment']
        
        npv_orig = calculate_npv(cf_orig, orig_rate)
        npv_new = calculate_npv(cf_new, new_rate)
        
        delta_npv = npv_new - npv_orig
        is_material = abs(delta_npv) > MATERIALITY_THRESHOLD

        npv_results.append({
            'loan_id': loan_id,
            'NPV_orig': npv_orig,
            'NPV_new': npv_new,
            'Delta_NPV': delta_npv,
            'material': is_material
        })
    
    result_df = pd.DataFrame(npv_results)
    return result_df

def run_sensitivity(df_master: pd.DataFrame, rate_shift_bp: int) -> pd.DataFrame:
    """
    Performs sensitivity analysis on NPV by shifting discount rates.
    rate_shift_bp: integer, basis points to shift the discount rate (e.g., 50 for +50bp, -100 for -100bp).
    """
    if df_master.empty:
        return pd.DataFrame()

    rate_shift_decimal = rate_shift_bp / 10000.0 # Convert basis points to decimal

    sensitivity_results = []

    for loan_id in df_master['loan_id'].unique():
        loan_data = df_master[df_master['loan_id'] == loan_id].copy()

        # Get the original and new discount rates for the loan
        orig_rate_base = loan_data['original_discount_rate'].iloc[0]
        new_rate_base = loan_data['new_discount_rate'].iloc[0]

        # Apply the rate shift
        orig_rate_shifted = orig_rate_base + rate_shift_decimal
        new_rate_shifted = new_rate_base + rate_shift_decimal

        # Prepare cash flows (using the same cash flows as before, only rates change)
        cf_orig = loan_data[loan_data['scenario'] == 'Original'].set_index('payment_number')['total_payment']
        cf_new = loan_data[loan_data['scenario'] == 'New'].set_index('payment_number')['total_payment']

        # Calculate new NPVs with shifted rates
        npv_orig_shifted = calculate_npv(cf_orig, orig_rate_shifted)
        npv_new_shifted = calculate_npv(cf_new, new_rate_shifted)
        delta_npv_shifted = npv_new_shifted - npv_orig_shifted

        sensitivity_results.append({
            'loan_id': loan_id,
            'Original_Rate_Shifted': orig_rate_shifted,
            'New_Rate_Shifted': new_rate_shifted,
            'NPV_orig_Shifted': npv_orig_shifted,
            'NPV_new_Shifted': npv_new_shifted,
            'Delta_NPV_Shifted': delta_npv_shifted
        })

    return pd.DataFrame(sensitivity_results)
