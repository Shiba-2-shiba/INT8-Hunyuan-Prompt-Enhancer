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

    @patch("core.assets._read_source_marker")
    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("core.assets.snapshot_download")
    def test_ensure_model_exists(self, mock_download, mock_listdir, mock_exists, mock_read_marker):
        """Test that nothing happens if model exists and is not empty"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["model.onnx"] # Directory not empty
        mock_read_marker.return_value = "tencent/Hunyuan-PromptEnhancer"
        
        path = assets.ensure_model(self.conf)
        
        # Verify download NOT called
        mock_download.assert_not_called()
        
        # Verify path
        root = assets.get_root_path()
        expected = os.path.abspath(os.path.join(root, "models/prompt_enhancer"))
        self.assertEqual(path, expected)

    @patch("core.assets._write_source_marker")
    @patch("core.assets._read_source_marker")
    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("core.assets.snapshot_download")
    def test_ensure_model_downloads_if_missing(self, mock_download, mock_listdir, mock_exists, mock_read_marker, mock_write_marker):
        """Test that download is triggered if directory missing"""
        mock_exists.return_value = False
        mock_read_marker.return_value = ""
        
        path = assets.ensure_model(self.conf)
        
        # Verify download called
        mock_download.assert_called_with(
            repo_id="tencent/Hunyuan-PromptEnhancer",
            local_dir=os.path.join(assets.get_root_path(), "models/prompt_enhancer"),
            cache_dir=os.path.join(assets.get_root_path(), ".hf_home", "hub"),
            allow_patterns=None
        )
        mock_write_marker.assert_called_once()

    @patch("core.assets._write_source_marker")
    @patch("core.assets._read_source_marker")
    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("core.assets.snapshot_download")
    def test_ensure_model_downloads_if_empty(self, mock_download, mock_listdir, mock_exists, mock_read_marker, mock_write_marker):
        """Test that download is triggered if directory exists but empty"""
        mock_exists.return_value = True
        mock_listdir.return_value = [] # Empty list
        mock_read_marker.return_value = "tencent/Hunyuan-PromptEnhancer"
        
        path = assets.ensure_model(self.conf)
        
        # Verify download called
        mock_download.assert_called()
        mock_write_marker.assert_called_once()

    @patch("core.assets._write_source_marker")
    @patch("core.assets._read_source_marker")
    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("core.assets.snapshot_download")
    def test_ensure_model_downloads_if_repo_changed(self, mock_download, mock_listdir, mock_exists, mock_read_marker, mock_write_marker):
        """Model directory exists but source repo changed -> refresh."""
        mock_exists.return_value = True
        mock_listdir.return_value = ["config.json"]
        mock_read_marker.return_value = "old/repo"

        assets.ensure_model(self.conf)

        mock_download.assert_called_once()
        mock_write_marker.assert_called_once()

if __name__ == '__main__':
    unittest.main()
