import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import importlib

# Add repo root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies
sys.modules["core.config"] = MagicMock()
sys.modules["core.assets"] = MagicMock()
sys.modules["core.tokenizer_patch"] = MagicMock()
sys.modules["core.cache"] = MagicMock()
sys.modules["core.prompts"] = MagicMock()
sys.modules["core.model"] = MagicMock()

# Do NOT mock core.postprocess
if "core.postprocess" in sys.modules:
    del sys.modules["core.postprocess"]

sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()

import nodes
importlib.reload(nodes)

class TestNodesTokenLogic(unittest.TestCase):

    def setUp(self):
        self.node = nodes.INT8_Hunyuan_PromptEnhancer()
        
        # Patch config
        self.policy_patcher = patch("nodes.config.load_policy")
        self.mock_load_policy = self.policy_patcher.start()
        self.mock_load_policy.return_value = {"banned_words": {}}
        
        self.config_patcher = patch("nodes.config.load_config")
        self.mock_load_config = self.config_patcher.start()
        self.mock_load_config.return_value = {"int8": {}}
        
        # Mocks
        self.mock_cache = sys.modules["core.cache"]
        self.mock_enhancer = MagicMock()
        self.mock_cache.get_enhancer.return_value = self.mock_enhancer
        self.mock_enhancer.predict.return_value = "Enhanced Prompt"

    def tearDown(self):
        self.policy_patcher.stop()
        self.config_patcher.stop()

    def test_token_doubling_with_thinking(self):
        """Test that max_new_tokens is doubled when thinking is enabled"""
        self.node.run(
            text="foo", style_policy="illustration (Tag List)", 
            temperature=0.7, top_p=0.9, 
            max_new_tokens=512, 
            seed=123, 
            enable_thinking=True, # ON
            device_map="auto", attn_backend="auto"
        )
        
        args, _ = self.mock_enhancer.predict.call_args
        config = args[2]
        # 512 * 2 = 1024
        self.assertEqual(config['max_new_tokens'], 1024)

    def test_token_cap_at_8192(self):
        """Test that max_new_tokens is capped at 8192"""
        self.node.run(
            text="foo", style_policy="illustration (Tag List)", 
            temperature=0.7, top_p=0.9, 
            max_new_tokens=4096, 
            seed=123, 
            enable_thinking=True, # ON
            device_map="auto", attn_backend="auto"
        )
        
        args, _ = self.mock_enhancer.predict.call_args
        config = args[2]
        # 4096 * 2 = 8192 (Just hits cap)
        self.assertEqual(config['max_new_tokens'], 8192)

    def test_token_no_doubling_without_thinking(self):
        """Test that max_new_tokens is NOT changed when thinking is disabled"""
        self.node.run(
            text="foo", style_policy="illustration (Tag List)", 
            temperature=0.7, top_p=0.9, 
            max_new_tokens=512, 
            seed=123, 
            enable_thinking=False, # OFF
            device_map="auto", attn_backend="auto"
        )
        
        args, _ = self.mock_enhancer.predict.call_args
        config = args[2]
        self.assertEqual(config['max_new_tokens'], 512)

if __name__ == '__main__':
    unittest.main()
