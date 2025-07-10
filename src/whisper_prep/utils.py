import argparse
from pathlib import Path
import zlib
import re
from fastlid import fastlid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=config_path)
    return parser.parse_args()


def config_path(path: str):
    if Path(path).exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"Config path:{path} is not a valid path.")


def get_compression_ratio(text: str):
    compression_ratio = len(text) / len(zlib.compress(text.encode("utf-8")))
    return compression_ratio


# ---------------------------------------------------------------------------
# 1. compile once – these are the micro-second French heuristics
# ---------------------------------------------------------------------------
TAG_RE = re.compile(r"<\|[^|]+\|>")  # strip <|0.16|> etc.

FR_WORDS = (
    r"\b(?:que|qui|dans|avec|pour|chez|entre|sans|leur(?:s)?|une?|des?|du|la|le|les"
    r"|il|elle|ils|elles|ça|est|sera|sont)\b"
)

FR_APOS = r"\b(?:[ldcjtmnsq]\s*'|qu\s*')"  # l' d' qu' …

HEURISTIC_RE = re.compile(f"(?:{FR_WORDS}|{FR_APOS})", re.I)

# ---------------------------------------------------------------------------
# 2. fastlid setup – keep only fr & de to speed things up
# ---------------------------------------------------------------------------
fastlid.set_languages = ["fr", "de"]  # restrict search space


# ---------------------------------------------------------------------------
# 3. the combined detector
# ---------------------------------------------------------------------------
def is_french(line: str, prob_th: float = 0.85) -> bool:
    """
    If regexes matches, double check with model.
    """
    # normalise & de-tag
    txt = TAG_RE.sub("", line).replace("’", "'").strip()

    # 1️⃣ ultra-cheap lexical heuristic
    if HEURISTIC_RE.search(txt):
        try:
            lang, prob = fastlid(txt)
        except (IndexError, ValueError):  # no label came back
            return False
        return lang == "fr" and prob >= prob_th

    return False
