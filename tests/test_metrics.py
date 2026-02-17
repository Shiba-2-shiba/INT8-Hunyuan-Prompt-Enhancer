import unittest
import tempfile
import shutil
import os
import json
from unittest.mock import patch, MagicMock
import sys

# Add repo root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metrics import MetricsCollector

class TestMetricsCollector(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.collector = MetricsCollector()
        # Reset singleton state
        self.collector.enabled = False
        self.collector.log_file = None

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        self.collector.enabled = False
        self.collector.log_file = None

    def test_configure_disabled(self):
        """Test default disabled state"""
        config = {"metrics": {"enabled": False}}
        self.collector.configure(config)
        self.assertFalse(self.collector.enabled)

    def test_configure_enabled(self):
        """Test enabling metrics"""
        log_path = os.path.join(self.test_dir, "test_metrics.jsonl")
        config = {"metrics": {"enabled": True, "log_file": log_path}}
        
        self.collector.configure(config)
        self.assertTrue(self.collector.enabled)
        self.assertEqual(self.collector.log_file, log_path)

    def test_log_writes_file(self):
        """Test that log method writes JSONL"""
        log_path = os.path.join(self.test_dir, "test_metrics.jsonl")
        config = {"metrics": {"enabled": True, "log_file": log_path}}
        self.collector.configure(config)
        
        self.collector.log("test_event", {"foo": "bar"})
        
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r", encoding="utf-8") as f:
            line = f.readline()
            data = json.loads(line)
            self.assertEqual(data["event"], "test_event")
            self.assertEqual(data["foo"], "bar")
            self.assertIn("timestamp", data)

    def test_log_fail_open(self):
        """Test that logging failure does not crash app"""
        log_path = os.path.join(self.test_dir, "test_metrics.jsonl")
        config = {"metrics": {"enabled": True, "log_file": log_path}}
        self.collector.configure(config)
        
        # Simulate OS error on open
        with patch("builtins.open", side_effect=OSError("Disk full")):
            # Should NOT raise exception
            try:
                self.collector.log("test_event", {})
            except OSError:
                self.fail("MetricsCollector raised exception on file error (should fail-open)")

if __name__ == '__main__':
    unittest.main()
