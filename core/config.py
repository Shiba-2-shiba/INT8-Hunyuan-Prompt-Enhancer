import os
import omegaconf

def load_config():
    """
    Loads the ckpts.yaml configuration file.
    """
    script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_directory, 'ckpts.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    return omegaconf.OmegaConf.load(config_path)
