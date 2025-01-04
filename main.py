import time
import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from neuralforecast import NeuralForecast
from modules.cryptollm import CRYPTOLLM

# Streamlit Interface
st.set_page_config(page_title="Cryptocurrency Price Forecasting", layout="wide")

# Title and Description
st.title("Cryptocurrency Price Forecasting with LLMs")
st.markdown("""
This app uses a Large Language Model (TimeLLM2) to forecast daily cryptocurrency prices based on historical data.
""")

# Sidebar Inputs
st.sidebar.header("Configuration")

# Cryptocurrency Selection
crypto_list = {
    'Bitcoin (BTC)': 'BTC-USD',
    'Ethereum (ETH)': 'ETH-USD',
    'Binance Coin (BNB)': 'BNB-USD',
    'Cardano (ADA)': 'ADA-USD',
    'Solana (SOL)': 'SOL-USD'
}

crypto_name = st.sidebar.selectbox("Select Cryptocurrency", list(crypto_list.keys()))
crypto_symbol = crypto_list[crypto_name]

# Input fields for data selection
start_date = st.sidebar.date_input("Start Date", dt.date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date(2024, 10, 26))
forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 60, 30)
input_size = st.sidebar.slider("Input Size (days)", 30, 120, 50)

# Forecast Button
if st.sidebar.button("Start Forecasting"):
    with st.spinner("Fetching data and training the model..."):
        # Load Data
        st.subheader(f"Historical Data for {crypto_name}")
        data = yf.download(crypto_symbol, start=start_date, end=end_date, interval='1d')
        
        if data.empty:
            st.error("No data available for the selected date range. Please adjust the start and end dates.")
        else:
            # Preprocess data
            data = data.reset_index()
            data = data[['Date', 'Close']]
            data.columns = ['ds', 'y']
            data['ds'] = pd.to_datetime(data['ds'])
            data['unique_id'] = crypto_symbol
            
            # Display data
            st.write(data.head())
            
            # Split Data
            train_size = int(len(data) * 0.8)
            Y_train_df = data[:train_size]
            Y_test_df = data[train_size:].reset_index(drop=True)
            
            # Initialize TimeLLM2 model
            prompt_prefix = f"The dataset contains daily prices for {crypto_name}. Seasonal and yearly trends may be present."
            timellm = CRYPTOLLM(
                h=forecast_horizon,
                input_size=input_size,
                prompt_prefix=prompt_prefix,
                batch_size=24,
                windows_batch_size=24
            )
            
            # Initialize NeuralForecast
            nf = NeuralForecast(
                models=[timellm],
                freq='D'
            )
            
            # Train Model
            nf.fit(df=Y_train_df, val_size=forecast_horizon)
            
            # Generate Forecasts
            forecasts = nf.predict(futr_df=Y_test_df)
            
            st.success("Forecasting completed!")
            
            # Plot Results
            st.subheader(f"Forecast Results for {crypto_name}")
            fig = go.Figure()

            # Add training data
            fig.add_trace(go.Scatter(
                x=Y_train_df['ds'],
                y=Y_train_df['y'],
                mode='lines',
                name='Training Data',
                line=dict(color='blue')
            ))

            # Add actual prices
            fig.add_trace(go.Scatter(
                x=Y_test_df['ds'],
                y=Y_test_df['y'],
                mode='lines',
                name='Actual Prices',
                line=dict(color='green')
            ))

            # Add forecasted prices
            fig.add_trace(go.Scatter(
                x=forecasts['ds'],
                y=forecasts['TimeLLM2'],
                mode='lines',
                name='Forecasted Prices',
                line=dict(color='orange', dash='dot')
            ))

            # Customize layout
            fig.update_layout(
                title=f"{crypto_name} Price Forecasting",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                legend_title="Legend",
                template="plotly_white"
            )

            # Display Plotly graph
            st.plotly_chart(fig, use_container_width=True)

            # Display forecast data
            st.subheader("Forecast Data")
            st.write(forecasts)

