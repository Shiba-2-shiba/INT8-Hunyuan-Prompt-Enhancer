# This file is part of a derivative work based on Tencent Hunyuan.
# See LICENSE.txt and NOTICE.txt for details.

import os
import json

import logging
from huggingface_hub import snapshot_download

_SOURCE_MARKER = ".hf_source_repo"


def get_root_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_hf_cache_env(root_path):
    """
    Use project-local Hugging Face cache to avoid permission issues on shared/global cache dirs.
    """
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        hf_home = os.path.join(root_path, ".hf_home")
        os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _marker_path(local_dir):
    return os.path.join(local_dir, _SOURCE_MARKER)


def _read_source_marker(local_dir):
    marker = _marker_path(local_dir)
    if not os.path.exists(marker):
        return ""
    try:
        with open(marker, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _write_source_marker(local_dir, repo_id):
    os.makedirs(local_dir, exist_ok=True)
    with open(_marker_path(local_dir), "w", encoding="utf-8") as f:
        f.write(str(repo_id or "").strip())


def _has_complete_base_model(local_dir):
    """
    Verify HF model files are complete enough for AutoModel.from_pretrained.
    """
    if not os.path.isdir(local_dir):
        return False
    if not os.path.exists(os.path.join(local_dir, "config.json")):
        return False
    if os.path.exists(os.path.join(local_dir, "model.safetensors")):
        return True

    index_path = os.path.join(local_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return False
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        shard_files = set(weight_map.values())
        if not shard_files:
            return False
        for shard in shard_files:
            if not os.path.exists(os.path.join(local_dir, shard)):
                return False
        return True
    except Exception:
        return False


def ensure_model(conf):
    """
    Ensures the model is present in the local directory.
    Downloads if missing or empty.
    Returns the absolute path to the model directory.
    """
    root_path = get_root_path()
    _ensure_hf_cache_env(root_path)
    local_dir_rel = conf['local_dir']
    local_dir = os.path.join(root_path, local_dir_rel)
    
    target_repo = conf["repo_id"]
    existing_repo = _read_source_marker(local_dir)
    should_download = (not os.path.exists(local_dir)) or (not os.listdir(local_dir))
    if existing_repo and existing_repo != target_repo:
        logging.info(
            f"Model source changed ({existing_repo} -> {target_repo}). Refreshing local snapshot."
        )
        should_download = True
    if not should_download and not _has_complete_base_model(local_dir):
        logging.info("Model directory is incomplete. Refreshing local snapshot.")
        should_download = True

    if should_download:
        logging.info(f"Downloading Hunyuan Enhancer model to {local_dir}...")
        download_kwargs = dict(
            repo_id=target_repo,
            local_dir=local_dir,
            cache_dir=os.environ.get("HF_HUB_CACHE"),
            allow_patterns=[f"{conf.get('subfolder', '')}/*"] if conf.get('subfolder') else None
        )
        if conf.get("revision"):
            download_kwargs["revision"] = conf["revision"]
        if conf.get("ignore_patterns"):
            download_kwargs["ignore_patterns"] = conf["ignore_patterns"]
        snapshot_download(**download_kwargs)
        _write_source_marker(local_dir, target_repo)
    
    return os.path.abspath(local_dir)
