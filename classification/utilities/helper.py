from pathlib import Path
import torch
import datetime
import os
from contextlib import contextmanager
import sys
import torch
import random
import numpy as np

@contextmanager
def log_stdout_to_file(filepath):
    # Save the current stdout and stderr so we can restore them later
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        # Open the log file in append mode
        with open(filepath, 'a') as file:
            # Define a helper function to handle the output
            def write_to_both(text):
                old_stdout.write(text)  # Write to standard output
                file.write(text)        # Write to file

            # Create a class that overrides the write and flush methods
            class SplitOutput:
                def write(self, message):
                    write_to_both(message)

                def flush(self):
                    old_stdout.flush()
                    file.flush()

            # Set this new output class as the current stdout and stderr
            sys.stdout = sys.stderr = SplitOutput()
            yield
    finally:
        # Restore the original stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
def create_directory_name():
    current_datetime = datetime.datetime.now()
    # Extract the date, hour, and minute parts
    current_date = current_datetime.date()
    current_hour = current_datetime.hour
    current_minute = current_datetime.minute
    directory_name = f'{current_date}_{current_hour}:{current_minute}'
    return directory_name

def create_target_dir(target_dir:str):
    dir_name = create_directory_name()
    # Create target directory
    target_dir_path = Path(target_dir)/dir_name
    target_dir_path.mkdir(parents=True, exist_ok=True)
    return target_dir_path

def create_dir(path:str, dir_name:str):
    target_dir_path = Path(path)/dir_name
    target_dir_path.mkdir(parents=True, exist_ok=True)
    return target_dir_path


def convert_to_three_channels(inputs):
    return inputs.repeat(1, 3, 1, 1) 


def set_seed(seed: int):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"[INFO]: Seed set to: {seed} for Python's built-in random module, NumPy and PyTorch.")
    if torch.cuda.is_available():
        print(f"[INFO]: PyTorch CUDA seeds set to: {seed}")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
     # Ensure that the deterministic algorithms are used
    torch.backends.cudnn.deterministic = True
    print("[INFO]: PyTorch cuDNN deterministic setting set to True for reproducibility.")
    torch.backends.cudnn.benchmark = False
    print("[INFO]: PyTorch cuDNN benchmark setting set to False for reproducibility.\n")

    
    
def get_num_correct(preds, labels):
    return torch.sum(preds == labels.data).item() 
    
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """    

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = Path(target_dir) / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    
