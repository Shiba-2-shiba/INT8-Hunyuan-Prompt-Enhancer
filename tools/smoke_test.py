import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add repo root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- MOCKING DEPENDENCIES START ---
# We must mock these BEFORE importing nodes or core modules
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["huggingface_hub"] = MagicMock()
sys.modules["triton"] = MagicMock()

# Mock omega conf only if not present, but usually it is.
# If we want to test real config loading, we should use real omegaconf.
try:
    import omegaconf
except ImportError:
    # Fallback mock if environment is broken
    sys.modules["omegaconf"] = MagicMock()

# --- MOCKING DEPENDENCIES END ---

    try:
        # 1. Import Nodes
        print("[1/4] Importing nodes...", flush=True)
        import nodes
        from core import config, metrics
        
        # 2. Setup Mocks for Run
        print("[2/4] Setting up runtime mocks...", flush=True)
        
        with patch("core.config.load_config", return_value={'int8': 'dummy_path'}), \
             patch("core.config.load_policy") as mock_load_policy, \
             patch("core.assets.ensure_model", return_value='dummy_ckpt_path'), \
             patch("core.tokenizer_patch.patch_tokenizer"), \
             patch("core.cache.get_enhancer") as mock_get_enhancer:
            
            # Setup Policy Mock (Real dict to avoid MagicMock truthy issues)
            mock_load_policy.return_value = {
                "banned_words": {
                    "illustration": ["photo", "8k", "masterpiece"]
                },
                "metrics": {
                    "enabled": False  # Explicitly disable to avoid side-effects
                }
            }
            
            # Setup Enhancer Mock
            mock_enhancer = MagicMock()
            # Return a prompt that triggers banning logic (contains "photo")
            # "An adorable kitten" should survive.
            mock_enhancer.predict.return_value = "A photo of a cat, 8k, masterpiece, An adorable kitten"
            mock_get_enhancer.return_value = mock_enhancer

            # 3. Instantiate Node
            print("[3/4] Instantiating INT8_Hunyuan_PromptEnhancer...", flush=True)
            if "INT8_Hunyuan_PromptEnhancer" not in nodes.NODE_CLASS_MAPPINGS:
                 print("ERROR: Node class not found in mappings.", flush=True)
                 sys.exit(1)
            
            node_cls = nodes.NODE_CLASS_MAPPINGS["INT8_Hunyuan_PromptEnhancer"]
            node = node_cls()
            
            # Debug Config
            print(f"DEBUG: Metrics Enabled: {metrics.collector.enabled}", flush=True)
            print(f"DEBUG: Log File: {metrics.collector.log_file}", flush=True)

            # 4. Run Execution
            print("[4/4] Executing run() method...", flush=True)
            style = "illustration (Tag List)"
            
            # Run
            result = node.run(
                text="test prompt",
                style_policy=style,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=256,
                seed=42,
                enable_thinking=True,
                device_map="auto",
                attn_backend="auto"
            )
            
            print(f"Result: {result}", flush=True)
            
            # Expect "photo", "8k", "masterpiece" to be removed
            # "A  of a cat, , " -> cleaned up to "A of a cat" roughly?
            # strip_unwanted_photo_style splits by comma.
            # "A photo of a cat", "8k", "masterpiece"
            # "A photo of a cat" -> banned regex hits "photo"?
            # "photo" is in banned list.
            # So "A photo of a cat" might be removed if regex matches partial?
            # banned_re = re.compile("|".join(banned_patterns))
            # If "photo" is in pattern, "A photo of a cat" matches.
            # So the first chunk is removed!
            # Result might be empty string?
            
            # Let's adjust return value to have a safe part.
            # "An illustration of a cat, photo, 8k"
            
            if "kitten" in str(result):
                print(">>> SUCCESS: Smoke test passed!", flush=True)
                sys.exit(0)
            else:
                print(f">>> FAILURE: Unexpected result: {result}", flush=True)
                sys.exit(1)

    except Exception as e:
        print(f">>> CRITICAL FAILURE: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
