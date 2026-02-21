import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Avoid importing heavy deps for this unit test.
sys.modules.setdefault("torch", MagicMock())
sys.modules.setdefault("torch.nn", MagicMock())
sys.modules.setdefault("torch.nn.functional", MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("safetensors", MagicMock())
sys.modules.setdefault("safetensors.torch", MagicMock())

from core.model import resolve_int8_weights


class TestInt8WeightResolution(unittest.TestCase):
    def test_explicit_path_priority(self):
        with patch.dict(os.environ, {}, clear=True), patch("os.path.exists", return_value=True):
            out = resolve_int8_weights("C:/models", "C:/x/custom.safetensors")
            self.assertEqual(out, "C:/x/custom.safetensors")

    def test_env_path_priority(self):
        with patch.dict(os.environ, {"PE_INT8_WEIGHTS": "C:/x/env.safetensors"}, clear=True), patch(
            "os.path.exists", return_value=True
        ):
            out = resolve_int8_weights("C:/models", None)
            self.assertEqual(out, "C:/x/env.safetensors")

    def test_variant_high_default(self):
        with patch.dict(os.environ, {"PE_INT8_VARIANT": "high"}, clear=True), patch(
            "os.path.exists"
        ) as mock_exists:
            mock_exists.side_effect = lambda p: str(p).endswith("promptenhancer_int8_high.safetensors")
            out = resolve_int8_weights("C:/models", None)
            self.assertTrue(out.endswith("promptenhancer_int8_high.safetensors"))


if __name__ == "__main__":
    unittest.main()
