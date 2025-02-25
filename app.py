import streamlit as st
import datetime as dt
from modules.data import load_and_preprocess_data
from modules.trainer import train_model
from modules.forecasting import generate_forecasts
from modules.visualization import plot_forecasts
from utils.config_manager import ConfigManager

# Streamlit Interface
st.set_page_config(page_title="Cryptocurrency Price Forecasting", layout="wide")

# Title and Description
st.title("Cryptocurrency Price Forecasting with LLMs")
st.markdown("""
This app uses a Large Language Model (TimeLLM2) to forecast daily cryptocurrency prices based on historical data.
""")

# Sidebar Inputs
st.sidebar.header("Configuration")

# Load configuration
config_manager = ConfigManager()
llm_params_config = config_manager.get_llm_params_config()
llm_params = llm_params_config.get('models', {})

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

# Model Selection
# The keys are displayed in the selectbox, and the value is returned.
# We use name:name to make both key and value the model name.
llm_list = list(llm_params.keys())

llm_name = st.sidebar.selectbox("Select Model", llm_list)

# Input fields for data selection
start_date = st.sidebar.date_input("Start Date", dt.date(2024, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date(2024, 10, 26))
forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 7, 60, 30)
input_size = st.sidebar.slider("Input Size (days)", 30, 120, 50)


model_config = config_manager.get_model_config()
prompt_prefix = f"The dataset contains daily prices for {crypto_name}. Seasonal and yearly trends may be present."

# Create placeholder for training progress
training_progress = st.empty()

# Forecast Button
if st.sidebar.button("Start Forecasting"):
    with st.spinner("Fetching data and training the model..."):
        # Load and preprocess data
        Y_train_df, Y_test_df = load_and_preprocess_data(crypto_symbol, start_date, end_date)

        if Y_train_df is None or Y_test_df is None:
            st.error("No data available for the selected date range. Please adjust the start and end dates.")
        else:
            # Display data
            st.subheader(f"Historical Data for {crypto_name}")
            st.write(Y_train_df.head())

            # Train model with progress display
            training_progress.text("Starting model training...")
            nf = train_model(Y_train_df, Y_test_df, forecast_horizon, input_size, prompt_prefix, model_config,
                           llm=llm_name, progress_callback=lambda msg: training_progress.text(f"Training Progress: {msg}"))

            # Generate forecasts
            training_progress.text("Generating forecasts...")
            forecasts = generate_forecasts(nf, Y_test_df)

            st.success("Forecasting completed!")
            training_progress.empty()  # Clear the progress display

            # Plot results
            plot_forecasts(Y_train_df, Y_test_df, forecasts, crypto_name)
