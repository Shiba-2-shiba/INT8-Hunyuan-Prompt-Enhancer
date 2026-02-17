import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add repo root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies
sys.modules["huggingface_hub"] = MagicMock()

# Now import assets
from core import assets

class TestAssets(unittest.TestCase):

    def setUp(self):
        # Configuration mimicking what nodes.py passes
        self.conf = {
            "local_dir": "models/prompt_enhancer",
            "repo_id": "tencent/Hunyuan-PromptEnhancer",
        }

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("core.assets.snapshot_download")
    def test_ensure_model_exists(self, mock_download, mock_listdir, mock_exists):
        """Test that nothing happens if model exists and is not empty"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["model.onnx"] # Directory not empty
        
        path = assets.ensure_model(self.conf)
        
        # Verify download NOT called
        mock_download.assert_not_called()
        
        # Verify path
        root = assets.get_root_path()
        expected = os.path.abspath(os.path.join(root, "models/prompt_enhancer"))
        self.assertEqual(path, expected)

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("core.assets.snapshot_download")
    def test_ensure_model_downloads_if_missing(self, mock_download, mock_listdir, mock_exists):
        """Test that download is triggered if directory missing"""
        mock_exists.return_value = False
        
        path = assets.ensure_model(self.conf)
        
        # Verify download called
        mock_download.assert_called_with(
            repo_id="tencent/Hunyuan-PromptEnhancer",
            local_dir=os.path.join(assets.get_root_path(), "models/prompt_enhancer"),
            local_dir_use_symlinks=False,
            allow_patterns=None
        )

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("core.assets.snapshot_download")
    def test_ensure_model_downloads_if_empty(self, mock_download, mock_listdir, mock_exists):
        """Test that download is triggered if directory exists but empty"""
        mock_exists.return_value = True
        mock_listdir.return_value = [] # Empty list
        
        path = assets.ensure_model(self.conf)
        
        # Verify download called
        mock_download.assert_called()

if __name__ == '__main__':
    unittest.main()
