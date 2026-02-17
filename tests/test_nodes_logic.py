import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import importlib

# Add repo root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies BEFORE importing nodes
# We use a helper to avoiding polluting other tests with permanent mocks if possible, 
# but for now we'll just mock the heavy/IO stuff.
sys.modules["core.config"] = MagicMock()
sys.modules["core.assets"] = MagicMock()
sys.modules["core.tokenizer_patch"] = MagicMock()
sys.modules["core.cache"] = MagicMock()
sys.modules["core.prompts"] = MagicMock()
sys.modules["core.model"] = MagicMock()

# Do NOT mock core.postprocess - we want to test integration with the real logic
# or at least not break test_postprocess.py
if "core.postprocess" in sys.modules:
    del sys.modules["core.postprocess"]

# Mock other heavy imports
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()

import nodes
importlib.reload(nodes) # Ensure nodes uses our fresh mocks

class TestINT8HunyuanPromptEnhancer(unittest.TestCase):

    def setUp(self):
        self.node_cls = nodes.INT8_Hunyuan_PromptEnhancer
        self.node = self.node_cls()
        
        # We need to ensure we are patching the config used BY nodes.py
        # Since nodes.py has 'from .core import config', nodes.config IS the module.
        # We can patch the function on the module directly.
        
        # Create a patcher for load_policy
        self.policy_patcher = patch("nodes.config.load_policy")
        self.mock_load_policy = self.policy_patcher.start()
        
        # Setup Config Policy
        self.mock_policy = {"banned_words": {"illustration": ["mock_banned"]}}
        self.mock_load_policy.return_value = self.mock_policy
        
        # We also need to mock load_config because it's called in run()
        self.config_patcher = patch("nodes.config.load_config")
        self.mock_load_config = self.config_patcher.start()
        self.mock_load_config.return_value = {
            "int8": {
                "local_dir": "models/prompt_enhancer",
                "repo_id": "tencent/Hunyuan-PromptEnhancer"
            }
        }
        
        # Verify other mocks
        self.mock_assets = sys.modules["core.assets"]
        self.mock_tokenizer_patch = sys.modules["core.tokenizer_patch"]
        self.mock_cache = sys.modules["core.cache"]
        self.mock_prompts = sys.modules["core.prompts"]
        
        # Setup Prompts
        self.mock_prompts.SYS_PROMPT_ILLUST_OPTIMIZED = "SYS_ILLUST"
        self.mock_prompts.SYS_PROMPT_PHOTO_OPTIMIZED = "SYS_PHOTO"

        # Setup Enhancer mock
        self.mock_enhancer = MagicMock()
        self.mock_cache.get_enhancer.return_value = self.mock_enhancer
        self.mock_enhancer.predict.return_value = "Enhanced Prompt"
        
        # Spy on postprocess
        from core import postprocess
        self.original_strip = postprocess.strip_unwanted_photo_style
        postprocess.strip_unwanted_photo_style = MagicMock(side_effect=self.original_strip)
        self.spy_strip = postprocess.strip_unwanted_photo_style

    def tearDown(self):
        self.policy_patcher.stop()
        self.config_patcher.stop()
        
        # Restore original function
        from core import postprocess
        postprocess.strip_unwanted_photo_style = self.original_strip

    def test_run_illustration_policy(self):
        """Test that illustration policy selects correct system prompt and calls postprocess with policy"""
        style = "illustration (Tag List)"
        self.node.run(
            text="foo", 
            style_policy=style, 
            temperature=0.7, top_p=0.9, max_new_tokens=256, seed=123, 
            enable_thinking=True, device_map="auto", attn_backend="auto"
        )
        
        # Verify System Prompt
        self.mock_enhancer.predict.assert_called()
        args, _ = self.mock_enhancer.predict.call_args
        self.assertEqual(args[1], "SYS_ILLUST")
        
        # Verify Postprocess called with expected banned words
        self.spy_strip.assert_called()
        args, kwargs = self.spy_strip.call_args
        self.assertEqual(args[0], "Enhanced Prompt")
        self.assertEqual(args[1], "foo")
        self.assertEqual(args[2], ["mock_banned"])
        self.assertIn("collector", kwargs)

    def test_run_photography_policy(self):
        """Test that photography policy selects correct system prompt and SKIPS postprocess stripping"""
        style = "photography (Detailed)"
        self.node.run(
            text="foo", 
            style_policy=style, 
            temperature=0.7, top_p=0.9, max_new_tokens=256, seed=123, 
            enable_thinking=True, device_map="auto", attn_backend="auto"
        )
        
        # Verify System Prompt
        args, _ = self.mock_enhancer.predict.call_args
        self.assertEqual(args[1], "SYS_PHOTO")
        
        # Verify Postprocess NOT called
        self.spy_strip.assert_not_called()

    def test_run_custom_sys_prompt(self):
        """Test that custom system prompt overrides policy"""
        style = "illustration (Tag List)"
        self.node.run(
            text="foo", 
            style_policy=style, 
            temperature=0.7, top_p=0.9, max_new_tokens=256, seed=123, 
            enable_thinking=True, device_map="auto", attn_backend="auto",
            custom_sys_prompt="CUSTOM_SYS"
        )
        
        # Verify System Prompt
        args, _ = self.mock_enhancer.predict.call_args
        self.assertEqual(args[1], "CUSTOM_SYS")
        
        # Postprocess should still be called because style is illustration
        # Check that collector was passed
        self.spy_strip.assert_called()
        _, kwargs = self.spy_strip.call_args
        self.assertIn("collector", kwargs)


if __name__ == '__main__':
    unittest.main()
