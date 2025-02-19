import plotly.graph_objects as go
import streamlit as st

def plot_forecasts(Y_train_df, Y_test_df, forecasts, crypto_name):
    """Generates and displays the Plotly forecast plot.

    Args:
        Y_train_df: Training data DataFrame.
        Y_test_df: Test data DataFrame.
        forecasts: Forecasts DataFrame.
        crypto_name: Name of the cryptocurrency.
    """
    fig = go.Figure()

    # Add training data
    fig.add_trace(go.Scatter(
        x=Y_train_df['ds'],
        y=Y_train_df['y'],
        mode='lines',
        name='Training Data',
        line=dict(color='black')
    ))

    # Add test data
    fig.add_trace(go.Scatter(
        x=Y_test_df['ds'],
        y=Y_test_df['y'],
        mode='lines',
        name='Test Data',
        line=dict(color='green')
    ))

    # Add forecast
    if forecasts is not None:
        fig.add_trace(go.Scatter(
            x=forecasts['ds'],
            y=forecasts['CRYPTOLLM'],
            mode='lines',
            name='Forecast',
            line=dict(color='orange')
        ))

    fig.update_layout(
        title=f"Forecast Results for {crypto_name}",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
