from neuralforecast import NeuralForecast
from modules.cryptollm import CRYPTOLLM

def train_model(Y_train_df, Y_test_df, forecast_horizon, input_size, prompt_prefix, config):
    """Trains the CRYPTOLLM model using NeuralForecast.

    Args:
        Y_train_df: Training data DataFrame.
        Y_test_df: Validation data DataFrame.
        forecast_horizon: Forecast horizon.
        input_size: Input size.
        prompt_prefix: Prompt prefix for the LLM.
        config: Model configuration dictionary.

    Returns:
        nf: Trained NeuralForecast object.
    """

    timellm = CRYPTOLLM(
        h=forecast_horizon,
        input_size=input_size,
        prompt_prefix=prompt_prefix,
        max_steps=config.get('max_steps', 30), # Get from config
        early_stop_patience_steps=config.get('early_stop_patience_steps', 10), # Get from config
        learning_rate=config.get('learning_rate', 1e-4), # Get from config
        val_check_steps=config.get('val_check_steps', 10), # Get from config
        batch_size = config.get('batch_size', 24),
        windows_batch_size = config.get('windows_batch_size', 24)
    )

    nf = NeuralForecast(
        models=[timellm],
        freq='D'
    )

    nf.fit(df=Y_train_df, val_size=forecast_horizon)
    # No need to return forecasts here, just the trained model
    return nf
