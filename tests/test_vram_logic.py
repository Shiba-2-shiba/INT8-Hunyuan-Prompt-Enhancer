import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock comfy and torch before importing nodes
sys.modules["comfy"] = MagicMock()
sys.modules["comfy.model_management"] = MagicMock()
# We need real torch or mocked torch that behaves somewhat correctly
# Attempting to mock torch might be tricky if nodes.py imports it inside run
# But nodes.py run method does "import torch". 
# If we run this in environment with torch, it calls real torch.
# If we mock it in sys.modules, the import inside function might get the mock.

# Mock core modules which are dependencies
mock_core = MagicMock()
sys.modules["core"] = mock_core
sys.modules["core.config"] = MagicMock()
sys.modules["core.assets"] = MagicMock()
sys.modules["core.tokenizer_patch"] = MagicMock()
sys.modules["core.cache"] = MagicMock()
sys.modules["core.prompts"] = MagicMock()
sys.modules["core.postprocess"] = MagicMock()
sys.modules["core.metrics"] = MagicMock()

# Determine where nodes.py is and import it
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nodes

class TestVRAMLogic(unittest.TestCase):
    @patch("comfy.model_management.soft_empty_cache")
    @patch("nodes.config.load_config")
    @patch("nodes.config.load_policy")
    @patch("nodes.assets.ensure_model")
    @patch("nodes.tokenizer_patch.patch_tokenizer")
    @patch("nodes.cache.get_enhancer")
    def test_run_vram_calls(self, mock_get_enhancer, mock_patch, mock_ensure, mock_policy, mock_config, mock_soft_empty):
        # Setup Mocks
        mock_config.return_value = {'int8': {}}
        mock_policy.return_value = {'banned_words': {}}
        
        mock_enhancer = MagicMock()
        mock_enhancer.predict.return_value = "enhanced prompt"
        # Mocking model.device behavior
        mock_device = MagicMock()
        mock_device.__str__.return_value = 'cpu'
        mock_enhancer.model.device = mock_device
        # Wait, str(mock) returns string representation of mock object, not 'cpu' unless we configure __str__
        # Better:
        type(mock_enhancer.model).device = MagicMock(__str__=lambda x: 'cpu')
        # Actually simplest is just allow it to be anything or set side_effect for str() 
        # But let's just assert offload is called.
        
        mock_get_enhancer.return_value = mock_enhancer
        
        node = nodes.INT8_Hunyuan_PromptEnhancer()
        
        # Execute
        try:
            result = node.run(
                text="test", 
                style_policy="illustration (Tag List)", 
                temperature=0.0, 
                top_p=0.9, 
                top_k=5, 
                max_new_tokens=100, 
                seed=42, 
                enable_thinking=False, 
                device_map="cpu", # Use cpu to avoid torch.cuda checks needed
                attn_backend="auto"
            )
        except Exception as e:
            print(f"Execution failed: {e}")
            raise e
        
        # Verify
        print(f"soft_empty_cache call count: {mock_soft_empty.call_count}")
        
        # Check start call
        # soft_empty_cache should be called
        self.assertTrue(mock_soft_empty.called)
        
        # Check offload
        mock_enhancer.offload.assert_called_once()
        print("offload called successfully")

if __name__ == "__main__":
    unittest.main()
