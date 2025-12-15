"""Tests for Netflix-style SRT normalization."""

import unittest
import pysubs2
from whisper_prep.utils import fuse_until_limits


class TestFuseUntilLimits(unittest.TestCase):
    """Test the fuse_until_limits function."""

    def test_basic_fusion_within_limits(self):
        """Test that segments are fused when within duration and character limits."""
        subs = pysubs2.SSAFile()
        # Segment 1: 1-4 seconds (3 seconds), "Hello" (5 chars)
        subs.append(pysubs2.SSAEvent(start=1000, end=4000, text="Hello"))
        # Segment 2: 4-8 seconds (4 seconds total: 1-8 is 7 seconds), "World" (5 chars)
        subs.append(pysubs2.SSAEvent(start=4000, end=8000, text="World"))
        # Total: 7 seconds, 11 chars (within limits)
        
        changed = fuse_until_limits(subs, max_chars=42, max_duration=7)
        
        self.assertTrue(changed, "Should have merged segments")
        self.assertEqual(len(subs), 1, "Should have 1 segment after fusion")
        self.assertEqual(subs[0].start, 1000)
        self.assertEqual(subs[0].end, 8000)
        self.assertIn("Hello", subs[0].text)
        self.assertIn("World", subs[0].text)

    def test_no_fusion_exceeds_duration(self):
        """Test that segments are NOT fused when duration would exceed limit."""
        subs = pysubs2.SSAFile()
        # Segment 1: 1-4 seconds (3 seconds), "Hello"
        subs.append(pysubs2.SSAEvent(start=1000, end=4000, text="Hello"))
        # Segment 2: 5-12 seconds (7 seconds, total would be 11 seconds > 7)
        subs.append(pysubs2.SSAEvent(start=5000, end=12000, text="World"))
        
        changed = fuse_until_limits(subs, max_chars=42, max_duration=7)
        
        self.assertFalse(changed, "Should not have merged segments")
        self.assertEqual(len(subs), 2, "Should have 2 segments")

    def test_no_fusion_exceeds_characters(self):
        """Test that segments are NOT fused when character count would exceed limit."""
        subs = pysubs2.SSAFile()
        # Segment 1: "Hello there, how are you today?" (31 chars)
        subs.append(pysubs2.SSAEvent(start=1000, end=2000, text="Hello there, how are you today?"))
        # Segment 2: "I am doing great, thank you very much!" (38 chars, total 70 chars > 42)
        subs.append(pysubs2.SSAEvent(start=2000, end=3000, text="I am doing great, thank you very much!"))
        
        changed = fuse_until_limits(subs, max_chars=42, max_duration=7)
        
        self.assertFalse(changed, "Should not have merged segments")
        self.assertEqual(len(subs), 2, "Should have 2 segments")

    def test_three_segments_with_filter_word(self):
        """Test three segments where middle one contains filter word - none should fuse."""
        subs = pysubs2.SSAFile()
        # Segment 1: 1-4 seconds
        subs.append(pysubs2.SSAEvent(start=1000, end=4000, text="Hello"))
        # Segment 2: 5-8 seconds (contains filter word)
        subs.append(pysubs2.SSAEvent(start=5000, end=8000, text="This has PII"))
        # Segment 3: 8-9 seconds
        subs.append(pysubs2.SSAEvent(start=8000, end=9000, text="World"))
        
        changed = fuse_until_limits(subs, max_chars=42, max_duration=7, skip_words=["PII"])
        
        self.assertFalse(changed, "Should not fuse any segments due to PII")
        self.assertEqual(len(subs), 3, "Should have 3 segments")

    def test_filter_word_prevents_fusion_with_neighbors(self):
        """Test that a segment with filter word doesn't get fused with anything."""
        subs = pysubs2.SSAFile()
        # Segment 1: 1-4 seconds (3 seconds)
        subs.append(pysubs2.SSAEvent(start=1000, end=4000, text="Hello"))
        # Segment 2: 4-5 seconds (1 second, but contains PII)
        subs.append(pysubs2.SSAEvent(start=4000, end=5000, text="PII data"))
        # Segment 3: 5-6 seconds (1 second)
        subs.append(pysubs2.SSAEvent(start=5000, end=6000, text="World"))
        
        changed = fuse_until_limits(subs, max_chars=42, max_duration=7, skip_words=["PII"])
        
        # No fusion should happen at all
        self.assertFalse(changed, "Should not fuse anything")
        self.assertEqual(len(subs), 3, "Should have 3 segments")

    def test_fusion_when_no_filter_words(self):
        """Test normal fusion behavior when no filter words are specified."""
        subs = pysubs2.SSAFile()
        # Segment 1: 1-4 seconds (3 seconds)
        subs.append(pysubs2.SSAEvent(start=1000, end=4000, text="Hello"))
        # Segment 2: 4-8 seconds (4 seconds total 7 seconds)
        subs.append(pysubs2.SSAEvent(start=4000, end=8000, text="World"))
        
        changed = fuse_until_limits(subs, max_chars=42, max_duration=7, skip_words=[])
        
        self.assertTrue(changed, "Should fuse segments")
        self.assertEqual(len(subs), 1, "Should have 1 segment after fusion")

    def test_gap_between_segments(self):
        """Test segments with gaps between them."""
        subs = pysubs2.SSAFile()
        # Segment 1: 1-2 seconds
        subs.append(pysubs2.SSAEvent(start=1000, end=2000, text="Hello"))
        # Gap of 1 second
        # Segment 2: 3-8 seconds (5 seconds, total would be 7 seconds from start=1000 to end=8000)
        subs.append(pysubs2.SSAEvent(start=3000, end=8000, text="World"))
        
        changed = fuse_until_limits(subs, max_chars=42, max_duration=7)
        
        self.assertTrue(changed, "Should fuse segments even with gap")
        self.assertEqual(len(subs), 1, "Should have 1 segment")
        self.assertEqual(subs[0].start, 1000)
        self.assertEqual(subs[0].end, 8000)

    def test_case_insensitive_filter_words(self):
        """Test that filter words are case-insensitive."""
        subs = pysubs2.SSAFile()
        subs.append(pysubs2.SSAEvent(start=1000, end=2000, text="Hello"))
        subs.append(pysubs2.SSAEvent(start=2000, end=3000, text="This has pii in lowercase"))
        subs.append(pysubs2.SSAEvent(start=3000, end=4000, text="World"))
        
        changed = fuse_until_limits(subs, max_chars=42, max_duration=7, skip_words=["PII"])
        
        self.assertFalse(changed, "Should not fuse due to case-insensitive PII match")
        self.assertEqual(len(subs), 3, "Should have 3 segments")

    def test_empty_subs(self):
        """Test with empty subtitle file."""
        subs = pysubs2.SSAFile()
        
        changed = fuse_until_limits(subs, max_chars=42, max_duration=7)
        
        self.assertFalse(changed, "Should not change empty file")
        self.assertEqual(len(subs), 0)

    def test_single_segment(self):
        """Test with single segment (no fusion possible)."""
        subs = pysubs2.SSAFile()
        subs.append(pysubs2.SSAEvent(start=1000, end=8000, text="Hello"))
        
        changed = fuse_until_limits(subs, max_chars=42, max_duration=7)
        
        self.assertFalse(changed, "Should not change single segment")
        self.assertEqual(len(subs), 1)


if __name__ == "__main__":
    unittest.main()
