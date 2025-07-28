"""
Unit tests for text_normalizer functions such as collapse_text,
standardize_text, normalize_abbrv, and normalize_text wrapper.
"""

import unittest

from whisper_prep.generation.text_normalizer import (
    remove_keywords_with_brackets,
    normalize_abbrv,
    standardize_text,
    normalize_capitalization,
    normalize_triple_dots,
    remove_bracketed_text,
    tokenize,
    collapse_ngrams,
    collapse_text,
    GermanNumberConverter,
    normalize_text,
)


class TestGenerate(unittest.TestCase):
    # remove_keywords_with_brackets removes bracketed segments containing keywords
    def test_remove_keywords_with_brackets(self):
        text = "<|tag|> Live-Untertitel content <|end|>"
        assert remove_keywords_with_brackets(text) == ""
        # no keywords present returns original text
        original = "<|tag|> no keyword here <|end|>"
        assert remove_keywords_with_brackets(original) == original

    # normalize_abbrv expands common German abbreviations to full form
    def test_normalize_abbrv(self):
        assert normalize_abbrv(" z.B. ") == " zum Beispiel "
        assert (
            normalize_abbrv("Das ist ein dumz.B.er Test.")
            == "Das ist ein dumz.B.er Test."
        )
        assert normalize_abbrv(" etc. ") == " et cetera "
        assert (
            normalize_abbrv("Heute könnten wir z.B. in die Bergwelt reisen.")
            == "Heute könnten wir zum Beispiel in die Bergwelt reisen."
        )
        # unknown text remains unchanged
        assert normalize_abbrv("abc") == "abc"

    # standardize_text removes unwanted symbols and collapses spaces
    def test_standardize_text(self):
        text = "Hello—World “quoted”"
        expected = 'Hello-World "quoted"'
        assert standardize_text(text) == expected

    # normalize_capitalization fixes words with exactly two uppercase letters and length>4
    def test_normalize_capitalization(self):
        assert normalize_capitalization(" HoUse Test ") == "House Test"
        # words not matching condition remain unchanged
        assert normalize_capitalization(" USA test ") == "USA test"

    # normalize_triple_dots normalizes '...' according to context
    def normalize_triple_dots(self):
        assert normalize_triple_dots("First...Second") == "First. Second"
        # other triple dots are removed
        assert normalize_triple_dots("Wait...") == "Wait"
        assert normalize_triple_dots("...") == "."
        assert normalize_triple_dots("Test ... ... Test") == "Test ... Test"

    # remove_bracketed_text removes text enclosed in square brackets
    def test_remove_bracketed_text(self):
        assert remove_bracketed_text("Keep [remove this] text") == "Keep  text"
        assert remove_bracketed_text("Keep (remove this) text") == "Keep  text"
        assert remove_bracketed_text("<font color='#fff'>Jauchzer</font>") == "Jauchzer"
        assert remove_bracketed_text("Keep <font color='#fff'>Jauchzer</font> text") == "Keep Jauchzer text"

    # tokenize splits text into tokens, treating '...' as standalone token
    def test_tokenize(self):
        assert tokenize("one two ... three") == ["one", "two", "...", "three"]

    # collapse_ngrams collapses repeated tokens and n-grams; collapse_text combines tokenize
    def test_collapse_ngrams_and_collapse_text(self):
        tokens = [
            "...",
            "Triage",
            "...",
            "Triage",
            "...",
            "Triage",
            "...",
            "Triage",
            "...",
            "Triage",
        ]
        assert collapse_ngrams(tokens) == ["...", "Triage"]
        assert collapse_text(" ".join(tokens)) == "... Triage"

        text = "Heute gehen wir an den See ... See ... See ... See ... See"
        assert collapse_text(text) == "Heute gehen wir an den See ... See"
        text = "Heute gehen wir an den See und Baden da! Jawohl, es ist sonnig."
        assert collapse_text(text) == text

    # GermanNumberConverter converts numbers, apostrophes, commas, and currency symbols
    def test_german_number_converter(self):
        converter = GermanNumberConverter()
        text = "150 000 3'000 1,234 $"
        assert converter.convert(text) == "150000 3000 1.234 Dollar"

    # normalize_text applies full normalization pipeline end-to-end
    def test_normalize_text_integration(self):
        text = "Hallo [remove]ABC... Test z.B. 150 000 €"
        result = normalize_text(text)
        assert result == "Hallo ABC. Test zum Beispiel 150000 Euro"


if __name__ == "__main__":
    unittest.main()
