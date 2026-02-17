import os
import omegaconf

# Default policy (Fallback if policy.yaml is missing)
DEFAULT_POLICY = {
    "banned_words": {
        "illustration": [
            r"photo(graphy)?", "photorealistic", "realism", "realistic",
            r"hyper[-\s]?realistic", r"hyper[-\s]?realism", r"ultra[-\s]?realistic",
            "cinematic", "dslr", "35mm", "lens", "dof", 
            "depth of field", "bokeh", r"render(ing)?", "8k", 
            "overall", "presentation", "masterpiece", "best quality",
            "写实", "摄影", "照片", "实写"
        ]
    }
}

def load_config():
    """
    Loads the ckpts.yaml configuration file.
    """
    script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(script_directory, 'ckpts.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    return omegaconf.OmegaConf.load(config_path)

def load_policy():
    """
    Loads policy.yaml. If not found, returns DEFAULT_POLICY.
    """
    script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    policy_path = os.path.join(script_directory, 'policy.yaml')
    
    if os.path.exists(policy_path):
        try:
            user_policy = omegaconf.OmegaConf.load(policy_path)
            # Merge with default to ensure structure exists
            # (Simple merge: user overrides default)
            default_conf = omegaconf.OmegaConf.create(DEFAULT_POLICY)
            merged = omegaconf.OmegaConf.merge(default_conf, user_policy)
            return merged
        except Exception as e:
            print(f"Warning: Failed to load policy.yaml ({e}). Using default policy.")
            return omegaconf.OmegaConf.create(DEFAULT_POLICY)
    
    return omegaconf.OmegaConf.create(DEFAULT_POLICY)
