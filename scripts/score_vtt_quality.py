"""
VTT quality scoring against pseudo-labeled SRTs.

Two-step filter:
  1. Global WER  > GLOBAL_THRESHOLD  → FAIL_GLOBAL  (skip)
  2. Any 120s segment WER > SEGMENT_THRESHOLD → FAIL_SEGMENT (flag)
  Otherwise: OK

Usage:
    python scripts/score_vtt_quality.py Arena "SRF bi de Lüt – Abenteuer Wildnis"
    (defaults to both if no args)
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

SRG_DIR = Path("/mnt/nas05/data01/vincenzo/SRG_data_v4")
OUT_DIR = Path("/home/vincenzo/whisper-prep/analysis/vtt_quality")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_THRESHOLD  = 0.20
SEGMENT_THRESHOLD = 0.50
SEGMENT_MS        = 120_000

# ── Normalization ──────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"</?[^>]+>", "", text)          # strip VTT/HTML tags
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r" +", " ", text).strip()

# ── WER (word-level Levenshtein / |ref|) ───────────────────────────────────────

def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    # single-row DP (O(n) space)
    d = list(range(len(h) + 1))
    for rw in r:
        nd = [d[0] + 1]
        for j, hw in enumerate(h, 1):
            nd.append(min(d[j] + 1, nd[-1] + 1, d[j - 1] + (0 if rw == hw else 1)))
        d = nd
    return d[-1] / len(r)

# ── Subtitle parser (.srt and .vtt) ───────────────────────────────────────────

def _ts_ms(ts: str) -> int:
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    h, m, s = (parts if len(parts) == 3 else ["0"] + parts)
    return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)

def parse_subtitle(path: Path) -> list[tuple[int, int, str]]:
    """→ [(start_ms, end_ms, normalized_text), ...]"""
    result, block, s0, e0, state = [], [], 0, 0, "seek"
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if state == "seek":
                if " --> " in line:
                    left, right = line.split(" --> ", 1)
                    s0, e0 = _ts_ms(left), _ts_ms(right.split()[0])
                    block, state = [], "text"
            elif state == "text":
                if line == "":
                    text = normalize(" ".join(block))
                    if text:
                        result.append((s0, e0, text))
                    state = "seek"
                elif not line.isdigit() and line != "WEBVTT":
                    block.append(line)
    if block:
        text = normalize(" ".join(block))
        if text:
            result.append((s0, e0, text))
    return result

def text_in_window(segs: list, t0: int, t1: int) -> str:
    return " ".join(t for s, e, t in segs if s < t1 and e > t0)

# ── Episode scoring ────────────────────────────────────────────────────────────

def score_episode(srt_path: Path, vtt_path: Path) -> dict | None:
    srt = parse_subtitle(srt_path)
    vtt = parse_subtitle(vtt_path)
    if not srt or not vtt:
        return None

    # Compute rolling WER in SEGMENT_MS windows (small DP each time, O(words_per_segment²))
    # Global WER = weighted average of segment WERs — avoids giant full-episode DP
    max_end = max(e for _, e, _ in srt + vtt)
    segments = []
    total_edits = total_ref_words = 0
    t = 0
    while t < max_end:
        ref_w = text_in_window(srt, t, t + SEGMENT_MS)
        hyp_w = text_in_window(vtt, t, t + SEGMENT_MS)
        if ref_w:
            r = ref_w.split()
            h = hyp_w.split() if hyp_w else []
            # reuse DP result for both segment WER and global accumulation
            d = list(range(len(h) + 1))
            for rw in r:
                nd = [d[0] + 1]
                for j, hw in enumerate(h, 1):
                    nd.append(min(d[j] + 1, nd[-1] + 1, d[j-1] + (0 if rw == hw else 1)))
                d = nd
            edits = d[-1]
            total_edits += edits
            total_ref_words += len(r)
            segments.append((t // 1000, edits / len(r)))
        t += SEGMENT_MS

    if not segments:
        return None

    g_wer = total_edits / total_ref_words  # weighted average = global WER

    if g_wer > GLOBAL_THRESHOLD:
        return {"global_wer": g_wer, "verdict": "FAIL_GLOBAL", "segments": segments}

    max_seg = max(w for _, w in segments)
    verdict = "FAIL_SEGMENT" if max_seg > SEGMENT_THRESHOLD else "OK"
    return {"global_wer": g_wer, "verdict": verdict, "segments": segments}

# ── Show analysis ──────────────────────────────────────────────────────────────

def analyze_show(show_name: str) -> list[dict]:
    folder = SRG_DIR / show_name
    episodes = [
        (mp3.stem, mp3.with_suffix(".srt"), mp3.with_suffix(".vtt"))
        for mp3 in sorted(folder.glob("*.mp3"))
        if mp3.with_suffix(".srt").exists() and mp3.with_suffix(".vtt").exists()
    ]
    print(f"\n{'─'*60}")
    print(f"{show_name}: {len(episodes)} episodes with both SRT+VTT")
    print(f"{'─'*60}")

    results = []
    for i, (stem, srt, vtt) in enumerate(episodes, 1):
        print(f"  [{i}/{len(episodes)}] {stem[:8]}…", end="\r", flush=True)
        r = score_episode(srt, vtt)
        if r:
            r["id"] = stem
            results.append(r)
    print()

    counts = {v: sum(1 for r in results if r["verdict"] == v)
              for v in ("OK", "FAIL_SEGMENT", "FAIL_GLOBAL")}
    print(f"  OK={counts['OK']}  FAIL_SEGMENT={counts['FAIL_SEGMENT']}  FAIL_GLOBAL={counts['FAIL_GLOBAL']}")
    return results

# ── Plotting ───────────────────────────────────────────────────────────────────

COLORS = {"OK": "#27ae60", "FAIL_SEGMENT": "#f39c12", "FAIL_GLOBAL": "#e74c3c"}

def plot_show(show_name: str, results: list[dict]) -> None:
    if not results:
        return
    safe = re.sub(r"[^\w]+", "_", show_name).strip("_")
    out = OUT_DIR / safe
    out.mkdir(exist_ok=True)

    results_by_gwer = sorted(results, key=lambda r: r["global_wer"])

    # ── Fig 1: Global WER bar chart ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, len(results) * 0.35 + 2), 5))
    bar_colors = [COLORS[r["verdict"]] for r in results_by_gwer]
    ax.bar(range(len(results_by_gwer)),
           [r["global_wer"] for r in results_by_gwer],
           color=bar_colors, edgecolor="none", width=0.9)
    ax.axhline(GLOBAL_THRESHOLD, color="#e74c3c", ls="--", lw=1.5,
               label=f"Global threshold ({GLOBAL_THRESHOLD:.0%})")
    ax.set_xticks([])
    ax.set_ylabel("Global WER (VTT vs SRT)")
    ax.set_xlabel(f"Episodes (n={len(results)}, sorted by WER)")
    ax.set_title(f"{show_name} — Global WER per episode")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    patches = [mpatches.Patch(color=v, label=k) for k, v in COLORS.items()]
    ax.legend(handles=patches + [ax.get_lines()[0]], fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "global_wer.png", dpi=150)
    plt.close(fig)
    print(f"  → {out / 'global_wer.png'}")

    # ── Fig 2: Rolling WER heatmap (episodes that passed global filter) ────────
    ok_results = [r for r in results_by_gwer if r["segments"]]
    if not ok_results:
        return

    max_segs = max(len(r["segments"]) for r in ok_results)
    mat = np.full((len(ok_results), max_segs), np.nan)
    for i, r in enumerate(ok_results):
        for j, (_, w) in enumerate(r["segments"]):
            mat[i, j] = w

    h = max(4, len(ok_results) * 0.45)
    w = max(8, max_segs * 0.9)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(ok_results)))
    ax.set_yticklabels(
        [f"{r['id'][:8]}  ({r['global_wer']:.0%})" for r in ok_results],
        fontsize=7
    )
    ax.set_xlabel(f"Segment index (each = {SEGMENT_MS//1000}s)")
    ax.set_title(f"{show_name} — Rolling WER per {SEGMENT_MS//1000}s segment\n"
                 f"(only episodes passing {GLOBAL_THRESHOLD:.0%} global filter)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    cbar.set_label("WER")
    # threshold line on colorbar
    cbar.ax.axhline(SEGMENT_THRESHOLD, color="black", lw=1.5, ls="--")
    fig.tight_layout()
    fig.savefig(out / "rolling_wer_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  → {out / 'rolling_wer_heatmap.png'}")

    # ── Fig 3: WER distribution histogram ─────────────────────────────────────
    all_seg_wers = [w for r in ok_results for _, w in r["segments"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist([r["global_wer"] for r in results], bins=20,
                 color="#3498db", edgecolor="white")
    axes[0].axvline(GLOBAL_THRESHOLD, color="#e74c3c", ls="--", lw=1.5,
                    label=f"{GLOBAL_THRESHOLD:.0%} threshold")
    axes[0].set_xlabel("Global WER")
    axes[0].set_ylabel("# Episodes")
    axes[0].set_title("Global WER distribution")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[0].legend()

    axes[1].hist(all_seg_wers, bins=30, color="#9b59b6", edgecolor="white")
    axes[1].axvline(SEGMENT_THRESHOLD, color="#e74c3c", ls="--", lw=1.5,
                    label=f"{SEGMENT_THRESHOLD:.0%} threshold")
    axes[1].set_xlabel(f"Segment WER ({SEGMENT_MS//1000}s windows)")
    axes[1].set_ylabel("# Segments")
    axes[1].set_title("Rolling WER distribution (episodes passing global filter)")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[1].legend()

    fig.suptitle(show_name, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "wer_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  → {out / 'wer_distributions.png'}")


# ── Main ───────────────────────────────────────────────────────────────────────

SHOWS = sys.argv[1:] or ["Arena", "SRF bi de Lüt – Abenteuer Wildnis"]

for show in SHOWS:
    results = analyze_show(show)
    plot_show(show, results)
