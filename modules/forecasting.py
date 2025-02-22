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
