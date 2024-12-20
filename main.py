import time
import datetime as dt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.utils import augment_calendar_df
from modules.cryptollm import CRYPTOLLM

# Fetch historical daily Bitcoin price data
data = yf.download('BTC-USD', start='2024-01-01', end='2024-10-26', interval='1d')

# Reset index to have a proper 'ds' column for date
data = data.reset_index()

# Select only the date and closing price
data = data[['Date', 'Close']]
data.columns = ['ds', 'y']  # Renaming columns to match the format

# Ensure that the 'ds' column is in the correct datetime format
data['ds'] = pd.to_datetime(data['ds'])
# Add a unique_id column for the dataset
data['unique_id'] = 'bitcoin'  # Set a unique identifier for the entire dataset

# Prepare the data for training and testing
train_size = int(len(data) * 0.8)  # 80% for training
Y_train_df = data[:train_size]
Y_test_df = data[train_size:].reset_index(drop=True)

# Calculate suitable input size
#max_input_size = len(Y_train_df) - 30  # 30 is the forecasting horizon; adjust if needed
#input_size = max(1, min(max_input_size, 60))  # Use min to ensure it's not more than available data
prompt_prefix = "The dataset contains data on daily Bitcoin prices. There is potential for yearly trends and seasonal patterns."
# Initialize the TimeLLM2 model
timellm = CRYPTOLLM(
    h=30,  # Forecasting horizon (adjust as needed)
    input_size=50,  # Adjust this input size as discussed earlier  #36
    #llm= "Qwen/Qwen2.5-0.5B-Instruct",
    #d_llm=896,
    #llm=llm,
    #llm_config=llm_config,
    #llm_tokenizer=llm_tokenizer,
    prompt_prefix=prompt_prefix,
    batch_size=24,
    windows_batch_size=24
)
# Initialize NeuralForecast
nf = NeuralForecast(
    models=[timellm],
    freq='D'  # Daily frequency
)

# Fit the model on training data
nf.fit(df=Y_train_df, val_size=30)  # Validate on last 12 data points

# Generate forecasts for the test dataset
forecasts = nf.predict(futr_df=Y_test_df)

# Visualize the results
plt.figure(figsize=(14, 7))
plt.plot(Y_train_df['ds'], Y_train_df['y'], label='Training Data', color='blue')
plt.plot(Y_test_df['ds'], Y_test_df['y'], label='Actual Prices', color='green')
plt.plot(forecasts['ds'], forecasts['TimeLLM2'], label='Forecasted Prices', color='orange', linestyle='--')
plt.title('Bitcoin Price Forecasting')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# Print the forecasts DataFrame for inspection
print(forecasts)