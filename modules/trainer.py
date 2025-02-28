from forecast_utils import NeuralForecast
from modules.cryptollm import CRYPTOLLM
import pytorch_lightning as pl
import torch

def check_gpu_status():
    """Check and return detailed GPU status information, printing to terminal"""
    print("\n=== GPU Status Check ===")
    info = {}
    
    # Basic CUDA availability
    info["cuda_available"] = torch.cuda.is_available()
    print(f"CUDA available: {info['cuda_available']}")
    
    if info["cuda_available"]:
        # Get device count and properties
        device_count = torch.cuda.device_count()
        info["device_count"] = device_count
        print(f"Number of CUDA devices: {device_count}")
        info["devices"] = []
        
        for i in range(device_count):
            device_props = torch.cuda.get_device_properties(i)
            device_info = {
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(device_props.total_memory / (1024**3), 2),
                "compute_capability": f"{device_props.major}.{device_props.minor}"
            }
            info["devices"].append(device_info)
            print(f"Device {i}: {device_info['name']}")
            print(f"  Memory: {device_info['total_memory_gb']} GB")
            print(f"  Compute Capability: {device_info['compute_capability']}")
            
        # Get current device
        info["current_device"] = torch.cuda.current_device()
        print(f"Current device: {info['current_device']}")
        
        # Check if CUDA initialization is actually working
        try:
            # Try a small tensor operation on GPU
            x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            y = x + x
            info["cuda_functional"] = True
            print("CUDA functional: Yes")
        except Exception as e:
            info["cuda_functional"] = False
            info["cuda_error"] = str(e)
            print(f"CUDA functional: No - {info['cuda_error']}")
    
    print("========================\n")
    return info

class UIProgressCallback(pl.callbacks.Callback):
    """A PyTorch Lightning callback that reports training progress to the UI."""
    
    def __init__(self, progress_callback=None):
        super().__init__()
        self.progress_callback = progress_callback
    
    def on_train_start(self, trainer, pl_module):
        if self.progress_callback:
            self.progress_callback("Training started")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Report progress after each batch during training."""
        if self.progress_callback and batch_idx % 10 == 0:  # Report every 10 batches to avoid UI flooding
            metrics = trainer.callback_metrics
            loss = metrics.get('train_loss', None)
            if loss is not None:
                self.progress_callback(f"Batch {batch_idx} | Loss: {loss:.4f}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        if self.progress_callback:
            metrics = trainer.callback_metrics
            loss = metrics.get('train_loss_epoch', None)
            val_loss = metrics.get('val_loss', None)
            
            status = f"Epoch {trainer.current_epoch}"
            if loss is not None:
                status += f" | Loss: {loss:.4f}"
            if val_loss is not None:
                status += f" | Val Loss: {val_loss:.4f}"
                
            self.progress_callback(status)
    
    def on_validation_start(self, trainer, pl_module):
        """Report when validation begins."""
        if self.progress_callback:
            self.progress_callback(f"Starting validation - Epoch {trainer.current_epoch}")
    
    def on_validation_end(self, trainer, pl_module):
        if self.progress_callback:
            metrics = trainer.callback_metrics
            val_loss = metrics.get('val_loss', None)
            if val_loss is not None:
                self.progress_callback(f"Validation Loss: {val_loss:.4f}")
    
    def on_train_end(self, trainer, pl_module):
        if self.progress_callback:
            self.progress_callback("Training completed!")
            
    def on_exception(self, trainer, pl_module, exception):
        """Report if an exception occurs during training."""
        if self.progress_callback:
            self.progress_callback(f"Error during training: {str(exception)}")

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
    
    # Check GPU status
    check_gpu_status()

    # Create the CRYPTOLLM model without passing a step_callback
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
    )

    if progress_callback:
        progress_callback("Initializing NeuralForecast...")
    
    # Create a UI callback for progress reporting
    ui_callback = UIProgressCallback(progress_callback)
    
    # Add the UI callback to the model's trainer_kwargs
    if 'trainer_kwargs' not in timellm.__dict__:
        timellm.trainer_kwargs = {}
    
    if 'callbacks' not in timellm.trainer_kwargs:
        timellm.trainer_kwargs['callbacks'] = []
    
    timellm.trainer_kwargs['callbacks'].append(ui_callback)

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
