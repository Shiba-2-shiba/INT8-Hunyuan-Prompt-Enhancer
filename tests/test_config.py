import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add repo root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure omegaconf is available or mocked
try:
    import omegaconf
except ImportError:
    # If not installed, we mock it entirely for these tests
    sys.modules["omegaconf"] = MagicMock()
    import omegaconf

from core import config

class TestConfig(unittest.TestCase):

    def setUp(self):
        # Create a temp directory to simulate repo root
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_policy_default_when_missing(self):
        """Test that load_policy returns default if policy.yaml is missing"""
        with patch("core.config.os.path.exists") as mock_exists:
            mock_exists.return_value = False
            
            # We need to make sure omegaconf.OmegaConf.create returns a dict-like object
            # if we are mocking it. If it's real, it just works.
            # If sys.modules['omegaconf'] is a mock, we need to configure it.
            
            if isinstance(omegaconf, MagicMock):
                 # Configure the mock to behave like a dict for the default policy
                 def side_effect(arg):
                     return arg
                 omegaconf.OmegaConf.create.side_effect = side_effect
            
            policy = config.load_policy()
            
            # policy should be the dict from DEFAULT_POLICY (if create just returned it) 
            # or an OmegaConf object.
            # If it's a real OmegaConf object, we access it like attributes or dict.
            # If it's a dict (from our mock side effect), same for dict access.
            
            # config.DEFAULT_POLICY is a dict.
            # We should verify content.
            if hasattr(policy, 'banned_words'):
                self.assertIn("illustration", policy.banned_words)
            else:
                 self.assertIn("banned_words", policy)
                 self.assertIn("illustration", policy['banned_words'])

    def test_load_policy_with_file(self):
        """Test loading a real yaml file"""
        with patch("core.config.os.path.exists") as mock_exists, \
             patch("core.config.omegaconf.OmegaConf") as mock_oc:
            
            mock_exists.return_value = True
            
            # Setup mock for load()
            mock_oc.load.return_value = {"banned_words": {"illustration": ["custom_ban"]}}
            
            # Setup mock for create()
            # It's called for default policy: OmegaConf.create(DEFAULT_POLICY)
            # transform dict to something mockable if needed, or just return dict
            mock_oc.create.side_effect = lambda x: x
            
            # Setup mock for merge()
            # OmegaConf.merge(default, user)
            # We simulate the merge result
            def merge_side_effect(a, b):
                # Simple dict merge for testing
                # a is default dict, b is user dict
                # deep merge implementation for test
                merged = a.copy()
                merged.update(b) # shallow update, but good enough if b has full structure
                # actually b has structure {"banned_words": ...}
                # a has {"banned_words": ...}
                # result of merge in OmegaConf overwrites.
                return b 
            mock_oc.merge.side_effect = merge_side_effect
            
            policy = config.load_policy()
            
            # Check content
            banned = policy['banned_words']['illustration']
            self.assertIn("custom_ban", banned)


if __name__ == '__main__':
    unittest.main()
