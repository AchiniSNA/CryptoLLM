from neuralforecast import NeuralForecast
from modules.cryptollm import CRYPTOLLM

def train_model(Y_train_df, Y_test_df, forecast_horizon, input_size, prompt_prefix, config, llm=None, progress_callback=None):
    """Trains the CRYPTOLLM model using NeuralForecast.

    Args:
        Y_train_df: Training data DataFrame.
        Y_test_df: Validation data DataFrame.
        forecast_horizon: Forecast horizon.
        input_size: Input size.
        prompt_prefix: Prompt prefix for the LLM.
        config: Model configuration dictionary.
        llm: The LLM model to use.
        progress_callback: Optional callback function to report training progress.

    Returns:
        nf: Trained NeuralForecast object.
    """

    # Create a custom callback for training progress
    def training_step_callback(step, loss, val_loss=None):
        if progress_callback:
            status = f"Step {step}"
            if loss is not None:
                status += f" | Loss: {loss:.4f}"
            if val_loss is not None:
                status += f" | Val Loss: {val_loss:.4f}"
            progress_callback(status)

    timellm = CRYPTOLLM(
        h=forecast_horizon,
        input_size=input_size,
        prompt_prefix=prompt_prefix,
        llm=llm,
        max_steps=config.get('max_steps', 30),
        early_stop_patience_steps=config.get('early_stop_patience_steps', 10),
        learning_rate=config.get('learning_rate', 1e-4),
        val_check_steps=config.get('val_check_steps', 10),
        batch_size=config.get('batch_size', 24),
        windows_batch_size=config.get('windows_batch_size', 24),
        step_callback=training_step_callback  # Add the callback
    )

    if progress_callback:
        progress_callback("Initializing NeuralForecast...")

    nf = NeuralForecast(
        models=[timellm],
        freq='D'
    )

    if progress_callback:
        progress_callback("Starting model training...")

    nf.fit(df=Y_train_df, val_size=forecast_horizon)
    
    if progress_callback:
        progress_callback("Training completed!")

    return nf
