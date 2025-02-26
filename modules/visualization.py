import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import plotly.express as px

def plot_forecasts(Y_train_df, Y_test_df, forecasts, crypto_name, error_df=None, mape=None, direction_accuracy=None):
    """Generates and displays the Plotly forecast plot.

    Args:
        Y_train_df: Training data DataFrame.
        Y_test_df: Test data DataFrame.
        forecasts: Forecasts DataFrame.
        crypto_name: Name of the cryptocurrency.
        error_df: DataFrame containing error metrics.
        mape: Mean Absolute Percentage Error.
        direction_accuracy: Direction prediction accuracy.
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
    
    # Display error metrics if available
    if mape is not None and direction_accuracy is not None:
        st.subheader("Forecast Error Metrics")
        
        # Create two columns for metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%", 
                     delta=f"{-mape:.2f}%", delta_color="inverse")
            
        with col2:
            st.metric("Direction Prediction Accuracy", f"{direction_accuracy:.2f}%",
                     delta=f"{direction_accuracy - 50:.2f}%", delta_color="normal")
        
        st.info("MAPE: Lower is better. Direction Accuracy: Higher is better.")
    
    # Display error dataframe and visualization if available
    if error_df is not None:
        st.subheader("Detailed Error Analysis")
        
        # Show the error dataframe
        with st.expander("View Error Data"):
            st.dataframe(error_df[['ds', 'y', 'forecast', 'percentage_error', 'direction_correct']])
        
        # Plot percentage errors over time
        fig_error = px.line(error_df, x='ds', y='percentage_error', 
                           title='Percentage Error Over Time',
                           labels={'ds': 'Date', 'percentage_error': 'Percentage Error (%)'},
                           template="plotly_white")
        
        st.plotly_chart(fig_error, use_container_width=True)
        
        # Create histogram of percentage errors
        fig_hist = px.histogram(error_df, x='percentage_error', nbins=20,
                               title='Distribution of Percentage Errors',
                               labels={'percentage_error': 'Percentage Error (%)'},
                               template="plotly_white")
        
        st.plotly_chart(fig_hist, use_container_width=True)
