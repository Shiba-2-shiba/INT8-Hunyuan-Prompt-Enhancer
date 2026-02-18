# This file is part of a derivative work based on Tencent Hunyuan.
# See LICENSE.txt and NOTICE.txt for details.

import os

import logging
from huggingface_hub import snapshot_download

def get_root_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def ensure_model(conf):
    """
    Ensures the model is present in the local directory.
    Downloads if missing or empty.
    Returns the absolute path to the model directory.
    """
    root_path = get_root_path()
    local_dir_rel = conf['local_dir']
    local_dir = os.path.join(root_path, local_dir_rel)
    
    if not os.path.exists(local_dir) or not os.listdir(local_dir):
        logging.info(f"Downloading Hunyuan Enhancer model to {local_dir}...")
        snapshot_download(
            repo_id=conf['repo_id'],
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=[f"{conf.get('subfolder', '')}/*"] if conf.get('subfolder') else None
        )
    
    return os.path.abspath(local_dir)
