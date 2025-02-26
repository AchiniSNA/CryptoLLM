def generate_forecasts(nf, Y_test_df):
    """Generates forecasts using the trained NeuralForecast model.

    Args:
        nf: Trained NeuralForecast object.
        Y_test_df: Test data DataFrame.

    Returns:
        forecasts: Forecasts DataFrame.
    """
    forecasts = nf.predict(futr_df=Y_test_df)
    return forecasts

def calculate_forecast_errors(Y_test_df, forecasts):
    """Calculates percentage errors between actual and forecasted values.

    Args:
        Y_test_df: Test data DataFrame with actual values.
        forecasts: Forecasts DataFrame with predicted values.

    Returns:
        DataFrame with original data, forecasts, and error metrics.
    """
    # Ensure the dataframes have the same index
    merged_df = Y_test_df.copy()
    
    # Add forecast values to the merged dataframe
    merged_df['forecast'] = forecasts['CRYPTOLLM'].values
    
    # Calculate absolute percentage error
    merged_df['percentage_error'] = 100 * abs(merged_df['y'] - merged_df['forecast']) / merged_df['y']
    
    # Calculate mean absolute percentage error (MAPE)
    mape = merged_df['percentage_error'].mean()
    
    # Add direction error (whether the forecast correctly predicted up/down movement)
    merged_df['prev_actual'] = merged_df['y'].shift(1)
    merged_df['actual_direction'] = (merged_df['y'] > merged_df['prev_actual']).astype(int)
    merged_df['forecast_direction'] = (merged_df['forecast'] > merged_df['prev_actual']).astype(int)
    merged_df['direction_correct'] = (merged_df['actual_direction'] == merged_df['forecast_direction']).astype(int)
    
    # Calculate direction accuracy
    direction_accuracy = merged_df['direction_correct'].mean() * 100
    
    return merged_df, mape, direction_accuracy
