"""
Estimate total audio hours per show and per language.
Strategy: ffprobe one MP3 per show to get bitrate, then
          sum all file sizes and compute hours = bytes*8 / bitrate.
"""
import json
import subprocess
from collections import defaultdict
from pathlib import Path

SRG_DIR = Path("/mnt/nas05/data01/vincenzo/SRG_data_v4")
TSV = SRG_DIR / "_whisper_prep_inputs/srg_v4_transcripts.tsv"

SHOW_LANG = {
    "telegiornale": "it",
    "Il Quotidiano": "it",
    "19h30": "fr",
    "Couleurs locales": "fr",
}

# Build show->language map and set of MP3 paths that have a subtitle from TSV
show_lang_from_tsv: dict[str, str] = {}
mp3s_with_subtitle: set[str] = set()
with open(TSV) as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 5:
            audio_path, lang, show = parts[1], parts[3], parts[4]
            show_lang_from_tsv[show] = lang
            mp3s_with_subtitle.add(audio_path)
SHOW_LANG.update(show_lang_from_tsv)


def get_bitrate(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True, timeout=15,
    )
    return int(json.loads(out.stdout)["format"]["bit_rate"])


results = []
for folder in sorted(SRG_DIR.iterdir()):
    if not folder.is_dir() or folder.name.startswith("_"):
        continue
    mp3s = [f for f in folder.glob("*.mp3") if str(f) in mp3s_with_subtitle]
    if not mp3s:
        continue
    try:
        bitrate = get_bitrate(mp3s[0])
    except Exception as e:
        print(f"  WARN {folder.name}: {e}")
        continue

    total_bytes = sum(f.stat().st_size for f in mp3s)
    hours = total_bytes * 8 / bitrate / 3600
    lang = SHOW_LANG.get(folder.name, "de")
    results.append((folder.name, lang, len(mp3s), hours, bitrate // 1000))

results.sort(key=lambda x: -x[3])

print(f"\n{'Show':<45} {'Lang':>4} {'Files':>6} {'Hours':>8} {'kbps':>6}")
print("─" * 76)
for show, lang, n, hours, kbps in results:
    print(f"{show:<45} {lang:>4} {n:>6} {hours:>8.1f}h  {kbps:>4}k")

lang_hours: dict[str, float] = defaultdict(float)
lang_files: dict[str, int] = defaultdict(int)
for _, lang, n, hours, _ in results:
    lang_hours[lang] += hours
    lang_files[lang] += n

print(f"\n{'Language':<10} {'Files':>6} {'Hours':>8}")
print("─" * 28)
total_h = sum(lang_hours.values())
for lang, hours in sorted(lang_hours.items(), key=lambda x: -x[1]):
    print(f"{lang:<10} {lang_files[lang]:>6} {hours:>8.1f}h")
print("─" * 28)
print(f"{'TOTAL':<10} {sum(lang_files.values()):>6} {total_h:>8.1f}h")
