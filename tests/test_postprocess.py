import unittest
import sys
import os

# Add repo root to sys.path to allow importing core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import postprocess

class TestPostProcess(unittest.TestCase):

    def test_extract_reprompt_standard(self):
        """Test extraction with 'Reprompt:' prefix"""
        text = "Thinking process...\nReprompt: A beautiful sunset"
        result, meta = postprocess._extract_reprompt(text)
        self.assertEqual(result, "A beautiful sunset")
        self.assertEqual(meta["method"], "reprompt_tag")

    def test_extract_reprompt_hyphen(self):
        """Test extraction with 'Re-prompt:' prefix"""
        text = "Thinking...\nRe-prompt: An apple"
        result, meta = postprocess._extract_reprompt(text)
        self.assertEqual(result, "An apple")
        self.assertEqual(meta["method"], "reprompt_tag_hyphen")

    def test_extract_reprompt_xml(self):
        """Test extraction with <answer> tags"""
        text = "<think>Hmm</think><answer>A cat</answer>"
        result, meta = postprocess._extract_reprompt(text)
        self.assertEqual(result, "A cat")
        self.assertEqual(meta["method"], "xml")

    def test_extract_reprompt_fallback(self):
        """Test fallback extracting by removing <think> blocks"""
        text = "<think>Some thought process</think>Just the prompt"
        result, meta = postprocess._extract_reprompt(text)
        self.assertEqual(result, "Just the prompt")
        self.assertEqual(meta["method"], "fallback")

    def test_extract_reprompt_multiline(self):
        """Test that extraction handles multiline content correctly"""
        text = "Reprompt: Line 1\nLine 2"
        result, _ = postprocess._extract_reprompt(text)
        self.assertEqual(result, "Line 1\nLine 2")

    def test_extract_stops_at_footer(self):
        """Test that extraction stops at known footers like 'User:' or 'Raw:'"""
        text = "Reprompt: The prompt\nUser: invalid"
        result, _ = postprocess._extract_reprompt(text)
        self.assertEqual(result, "The prompt")

    # Define test banned patterns
    TEST_BANNED = ["photorealistic", "8k", "masterpiece", "cinematic"]

    def test_strip_with_collector(self):
        """Test logging when banned words are removed"""
        from unittest.mock import MagicMock
        collector = MagicMock()
        
        original = "Draw a girl"
        reprompt = "A photorealistic girl, 8k"
        
        # banned: photorealistic, 8k. 
        # both removed. count = 2.
        
        postprocess.strip_unwanted_photo_style(reprompt, original, self.TEST_BANNED, collector=collector)
        
        collector.log.assert_called_once()
        args = collector.log.call_args
        self.assertEqual(args[0][0], "banned_removal")
        self.assertEqual(args[0][1]["count"], 2)

    def test_strip_banned_words_basic(self):
        """Test removing banned words in illustration mode"""
        original = "Draw a girl"
        reprompt = "A photorealistic girl, 8k, masterpiece"
        # photorealistic, 8k, masterpiece are banned
        result = postprocess.strip_unwanted_photo_style(reprompt, original, self.TEST_BANNED)
        self.assertEqual(result, "")

    def test_strip_banned_words_preservation(self):
        """Test that banned words are KEPT if user asked for them"""
        original = "I want a photorealistic image"
        reprompt = "photorealistic, 8k"
        result = postprocess.strip_unwanted_photo_style(reprompt, original, self.TEST_BANNED)
        self.assertEqual(result, "photorealistic, 8k")

    def test_strip_banned_words_partial(self):
        """Test partial removal"""
        original = "Draw a cat"
        reprompt = "A cute cat, photorealistic, cinematic lighting"
        # "A cute cat" -> keep
        # "photorealistic" -> drop
        # "cinematic lighting" -> drop usage of "cinematic"
        result = postprocess.strip_unwanted_photo_style(reprompt, original, self.TEST_BANNED)
        self.assertEqual(result, "A cute cat")

    
    # -------------------------------------------------------------------------
    # Fixes for Thinking Mode Stability
    # -------------------------------------------------------------------------

    def test_fallback_removes_answer_tags(self):
        """Test fallback should remove <answer> and </answer> tags even if regex extraction fails"""
        # Case where <answer> exists but regex might fail or it's just leftover
        text = "<think>Thinking...</think>\n<answer> Final prompt </answer>"
        # Expected: "Final prompt" or at least no tags
        result, meta = postprocess._extract_reprompt(text)
        self.assertNotIn("<answer>", result)
        self.assertNotIn("</answer>", result)
        self.assertIn("Final prompt", result)

    def test_fallback_removes_broken_answer_tags(self):
        """Test fallback should remove orphaned </answer> tags"""
        text = "<think>Thinking...</think>\nFinal prompt</answer>"
        result, meta = postprocess._extract_reprompt(text)
        self.assertNotIn("</answer>", result)
        self.assertEqual(result, "Final prompt")

    def test_reprompt_cutoff_innocent_usage(self):
        """Test Reprompt extraction should NOT stop at innocent usage of keywords like 'Prompt'"""
        text = "Reprompt: A girl holding a sign that says 'Prompt engineering is fun'."
        result, meta = postprocess._extract_reprompt(text)
        # Current logic might cut at 'Prompt'
        self.assertIn("engineering is fun", result)
        self.assertEqual(meta["method"], "reprompt_tag")

    def test_reprompt_cutoff_strict_footer(self):
        """Test Reprompt extraction SHOULD stop at actual headers"""
        text = "Reprompt: The stored prompt\nUser: Next command"
        result, meta = postprocess._extract_reprompt(text)
        self.assertEqual(result, "The stored prompt")

    def test_extract_reprompt_removes_reprompt_marker_in_fallback(self):
        """Test fallback should remove 'Reprompt:' if it was missed by the main regex"""
        text = "<answer>Reprompt: The prompt</answer>"
        result, meta = postprocess._extract_reprompt(text)
        self.assertNotIn("Reprompt:", result)
        self.assertEqual(result, "The prompt")

    def test_think_tag_cleanup(self):
        """Test that <think> tags are removed in fallback"""
        text = "<think> long thought process </think> The actual prompt"
        result, meta = postprocess._extract_reprompt(text)
        self.assertEqual(result.strip(), "The actual prompt")

if __name__ == '__main__':
    unittest.main()
