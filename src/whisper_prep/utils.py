import argparse
from pathlib import Path
import zlib
import re
from fastlid import fastlid
import pysubs2
from glob import glob
import os


NETFLIX_CHAR = 42
NETFLIX_DUR = 7


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
        return lang == "fr"  # and prob >= prob_th

    return False


def fuse_until_limits(
    subs: pysubs2.SSAFile,
    max_chars: int = NETFLIX_CHAR,
    max_duration: float = NETFLIX_DUR,
) -> bool:
    """Rewrite *subs* in‑place, merging cues while combined cue stays <= limits.

    Returns True if *subs* was modified.
    """
    if not subs:
        return False

    merged = pysubs2.SSAFile()
    current = subs[0].copy()

    for cue in subs[1:]:
        combined_text = f"{current.text.rstrip()} {cue.text.lstrip()}"
        combined_dur = (cue.end - current.start) / 1000  # ms → s

        within_limits = len(combined_text) <= max_chars and combined_dur <= max_duration

        if within_limits:
            # Safe to merge: extend current cue
            current.end = cue.end
            current.text = combined_text
        else:
            # Would exceed limits -> flush current and start anew
            merged.append(current)
            current = cue.copy()

    merged.append(current)

    # Detect change by comparing lengths or any differing fields
    changed = len(merged) != len(subs) or any(
        m.start != s.start or m.end != s.end or m.text != s.text
        for m, s in zip(merged, subs)
    )

    if changed:
        subs.clear()
        subs.extend(merged)
    return changed


def normalize_file(path: str) -> None:
    subs = pysubs2.load(path)
    if fuse_until_limits(subs):
        subs.save(path, format_="srt")  # overwrite in place
        print(f"Updated {(path)}")


def netflix_normalize_all_srts_in_folder(folder: str = ".") -> None:
    """One-liner helper: normalize all .srt files in *folder*."""
    for file in glob(os.path.join(folder, "*.srt")):
        normalize_file(file)
