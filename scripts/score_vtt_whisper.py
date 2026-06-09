"""
Score VTT quality by running Whisper on 30s audio chunks and comparing
to VTT text.

Why this is better than VTT-vs-SRT:
  - SRT (pseudo-labels) was made by Whisper already → comparing VTT to SRT
    is just comparing two Whisper runs, not measuring subtitle quality.
  - Here we transcribe the EXACT 30s audio window fresh and compare to
    the VTT subtitle for that same window.
  - CH German speakers → Standard German subtitles: both Whisper output
    AND the VTT are in Standard German, so WER is a fair semantic signal.

Usage:
    conda run -n whisper_prep python scripts/score_vtt_whisper.py
"""
import json
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import whisper

# ── Config ─────────────────────────────────────────────────────────────────────

SRG_DIR     = Path("/mnt/nas05/data01/vincenzo/SRG_data_v4")
OUT_DIR     = Path("/home/vincenzo/whisper-prep/analysis/vtt_quality")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL   = "small"
SEGMENT_S       = 30
MIN_VTT_WORDS   = 10        # skip windows with too little text (music, silence)
N_EPISODES      = 5         # episodes to sample per show
N_WINDOWS       = 8         # 30s windows to sample per episode
SEED            = 42

SHOWS = {
    "Arena":                          "Arena (live subtitles — expected bad)",
    "SRF bi de Lüt – Abenteuer Wildnis": "Abenteuer Wildnis (post-produced — expected good)",
}

# ── Text helpers ───────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"</?[^>]+>", "", text)
    text = re.sub(r"[^a-z0-9äöüß ]+", " ", text)
    return re.sub(r" +", " ", text).strip()

def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for rw in r:
        nd = [d[0] + 1]
        for j, hw in enumerate(h, 1):
            nd.append(min(d[j] + 1, nd[-1] + 1, d[j-1] + (0 if rw == hw else 1)))
        d = nd
    return d[-1] / len(r)

# ── VTT parser ─────────────────────────────────────────────────────────────────

def ts_ms(ts: str) -> int:
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    h, m, s = (parts if len(parts) == 3 else ["0"] + parts)
    return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)

def parse_vtt(path: Path) -> list[tuple[int, int, str]]:
    result, block, s0, e0, state = [], [], 0, 0, "seek"
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if state == "seek":
                if " --> " in line:
                    left, right = line.split(" --> ", 1)
                    s0, e0 = ts_ms(left), ts_ms(right.split()[0])
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

def vtt_text_in_window(segs: list, t0_ms: int, t1_ms: int) -> str:
    return " ".join(t for s, e, t in segs if s < t1_ms and e > t0_ms)

# ── Audio extraction ───────────────────────────────────────────────────────────

def extract_audio(mp3: Path, start_s: float, duration_s: float, out_wav: str) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start_s), "-t", str(duration_s),
         "-i", str(mp3), "-ar", "16000", "-ac", "1", out_wav],
        capture_output=True,
    )
    return r.returncode == 0 and Path(out_wav).stat().st_size > 1000

# ── Main scoring ───────────────────────────────────────────────────────────────

def score_show(show_name: str, model, rng: random.Random) -> list[dict]:
    folder = SRG_DIR / show_name
    episodes = [
        mp3 for mp3 in sorted(folder.glob("*.mp3"))
        if mp3.with_suffix(".vtt").exists()
    ]
    if not episodes:
        print(f"  No VTT episodes found for {show_name}")
        return []

    sampled_eps = rng.sample(episodes, min(N_EPISODES, len(episodes)))
    print(f"\n{'─'*60}")
    print(f"{show_name}: sampling {len(sampled_eps)}/{len(episodes)} episodes")
    print(f"{'─'*60}")

    all_results = []

    for ep_idx, mp3 in enumerate(sampled_eps, 1):
        vtt_path = mp3.with_suffix(".vtt")
        vtt_segs = parse_vtt(vtt_path)
        if not vtt_segs:
            continue

        # Build candidate windows: 30s chunks that have enough VTT text
        duration_ms = max(e for _, e, _ in vtt_segs)
        candidates = []
        t = 5_000  # skip first 5s (intros/titles)
        while t + SEGMENT_S * 1000 < duration_ms - 5_000:
            txt = vtt_text_in_window(vtt_segs, t, t + SEGMENT_S * 1000)
            if len(txt.split()) >= MIN_VTT_WORDS:
                candidates.append(t)
            t += SEGMENT_S * 1000  # non-overlapping

        if not candidates:
            continue

        windows = rng.sample(candidates, min(N_WINDOWS, len(candidates)))
        windows.sort()

        ep_results = []
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_wav = tf.name

        for win_ms in windows:
            vtt_txt = vtt_text_in_window(vtt_segs, win_ms, win_ms + SEGMENT_S * 1000)
            if not extract_audio(mp3, win_ms / 1000, SEGMENT_S, tmp_wav):
                continue

            try:
                result = model.transcribe(tmp_wav, language="de")
                whisper_txt = normalize(result["text"])
            except Exception as e:
                print(f"    whisper error: {e}")
                continue

            w = wer(vtt_txt, whisper_txt)
            ep_results.append({
                "episode": mp3.stem,
                "t_start_s": win_ms // 1000,
                "vtt_text": vtt_txt[:120],
                "whisper_text": whisper_txt[:120],
                "wer": w,
                "vtt_words": len(vtt_txt.split()),
                "whisper_words": len(whisper_txt.split()),
            })

        os.unlink(tmp_wav)

        if ep_results:
            avg = sum(r["wer"] for r in ep_results) / len(ep_results)
            print(f"  [{ep_idx}/{len(sampled_eps)}] {mp3.stem[:8]}…"
                  f"  n={len(ep_results)}  avg_WER={avg:.1%}")
            for r in ep_results:
                verdict = "✓" if r["wer"] < 0.40 else ("~" if r["wer"] < 0.60 else "✗")
                print(f"    {r['t_start_s']:5d}s  WER={r['wer']:.1%} {verdict}"
                      f"  vtt={r['vtt_words']}w  whisper={r['whisper_words']}w")
            all_results.extend(ep_results)

    return all_results

# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_comparison(results_by_show: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {"Arena": "#e74c3c", "SRF bi de Lüt – Abenteuer Wildnis": "#27ae60"}
    bins = np.linspace(0, 1.2, 25)

    # ── WER histograms ─────────────────────────────────────────────────────────
    ax = axes[0]
    for show, results in results_by_show.items():
        wers = [r["wer"] for r in results]
        label = show.split("–")[-1].strip() if "–" in show else show
        ax.hist(wers, bins=bins, alpha=0.6, color=colors.get(show, "gray"),
                label=f"{label} (n={len(wers)})", edgecolor="white", density=True)
    ax.axvline(0.40, color="orange", ls="--", lw=1.5, label="40% threshold")
    ax.axvline(0.60, color="red",    ls="--", lw=1.5, label="60% threshold")
    ax.set_xlabel("WER (Whisper vs VTT)")
    ax.set_ylabel("Density")
    ax.set_title("WER distribution per 30s segment")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=8)

    # ── CDF ────────────────────────────────────────────────────────────────────
    ax = axes[1]
    for show, results in results_by_show.items():
        wers = sorted(r["wer"] for r in results)
        cdf = np.arange(1, len(wers) + 1) / len(wers)
        label = show.split("–")[-1].strip() if "–" in show else show
        ax.plot(wers, cdf, color=colors.get(show, "gray"), lw=2, label=label)
    ax.axvline(0.40, color="orange", ls="--", lw=1.5)
    ax.axvline(0.60, color="red",    ls="--", lw=1.5)
    ax.set_xlabel("WER (Whisper vs VTT)")
    ax.set_ylabel("Fraction of segments below WER")
    ax.set_title("CDF — lower is better")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=8)

    # ── Word count ratio scatter ────────────────────────────────────────────────
    ax = axes[2]
    for show, results in results_by_show.items():
        ratios = [r["whisper_words"] / max(r["vtt_words"], 1) for r in results]
        wers   = [r["wer"] for r in results]
        label  = show.split("–")[-1].strip() if "–" in show else show
        ax.scatter(ratios, wers, alpha=0.5, color=colors.get(show, "gray"),
                   label=label, s=30)
    ax.axhline(0.40, color="orange", ls="--", lw=1)
    ax.axhline(0.60, color="red",    ls="--", lw=1)
    ax.axvline(1.0,  color="gray",   ls=":",  lw=1)
    ax.set_xlabel("Whisper words / VTT words  (ratio ≈ 1 = verbatim)")
    ax.set_ylabel("WER")
    ax.set_title("Word count ratio vs WER")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=8)

    fig.suptitle("VTT quality: Whisper transcription vs VTT text (30s segments)\n"
                 "CH German spoken → Standard German subtitles",
                 fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "whisper_vtt_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n→ {out}")

    # ── Per-show summary ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(results_by_show), figsize=(6 * len(results_by_show), 5),
                             sharey=True)
    if len(results_by_show) == 1:
        axes = [axes]

    thresholds = [0.40, 0.60]
    thresh_labels = ["< 40% (good)", "40–60% (marginal)", "> 60% (bad)"]
    thresh_colors = ["#27ae60", "#f39c12", "#e74c3c"]

    for ax, (show, results) in zip(axes, results_by_show.items()):
        by_ep: dict[str, list[float]] = {}
        for r in results:
            by_ep.setdefault(r["episode"][:8], []).append(r["wer"])

        ep_labels = list(by_ep.keys())
        ep_wers   = [by_ep[k] for k in ep_labels]
        positions = range(len(ep_labels))

        bp = ax.boxplot(ep_wers, positions=list(positions), widths=0.5,
                        patch_artist=True, showfliers=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(colors.get(show, "#3498db"))
            patch.set_alpha(0.6)

        for t, c in zip(thresholds, ["orange", "red"]):
            ax.axhline(t, color=c, ls="--", lw=1.5)

        ax.set_xticks(list(positions))
        ax.set_xticklabels(ep_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("WER (Whisper vs VTT)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        title = show.split("–")[-1].strip() if "–" in show else show
        ax.set_title(title)

    fig.suptitle("WER per episode (30s segments)", fontweight="bold")
    fig.tight_layout()
    out2 = OUT_DIR / "whisper_per_episode.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"→ {out2}")

# ── Print good/bad examples ────────────────────────────────────────────────────

def print_examples(results_by_show: dict[str, list[dict]]) -> None:
    print("\n" + "═" * 70)
    print("EXAMPLES")
    print("═" * 70)
    for show, results in results_by_show.items():
        title = show.split("–")[-1].strip() if "–" in show else show
        best  = sorted(results, key=lambda r: r["wer"])[:2]
        worst = sorted(results, key=lambda r: r["wer"])[-2:]
        print(f"\n── {title} ──")
        for label, items in [("BEST (low WER)", best), ("WORST (high WER)", worst)]:
            print(f"  {label}:")
            for r in items:
                print(f"    {r['episode'][:8]}  t={r['t_start_s']}s  WER={r['wer']:.1%}")
                print(f"      VTT:     {r['vtt_text'][:100]}")
                print(f"      Whisper: {r['whisper_text'][:100]}")

# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(SEED)
    rng = random.Random(SEED)

    print(f"Loading Whisper {WHISPER_MODEL}…")
    model = whisper.load_model(WHISPER_MODEL)
    print("Ready.\n")

    results_by_show: dict[str, list[dict]] = {}

    for show in SHOWS:
        results = score_show(show, model, rng)
        results_by_show[show] = results

        wers = [r["wer"] for r in results]
        if wers:
            print(f"\n  {show}")
            print(f"    n={len(wers)}  mean={np.mean(wers):.1%}  "
                  f"median={np.median(wers):.1%}  "
                  f"p25={np.percentile(wers,25):.1%}  "
                  f"p75={np.percentile(wers,75):.1%}")
            for thresh, label in [(0.40, "< 40%"), (0.60, "< 60%")]:
                frac = sum(1 for w in wers if w < thresh) / len(wers)
                print(f"    {label}: {frac:.0%} of segments")

    # Save raw results
    out_json = OUT_DIR / "whisper_results.json"
    with open(out_json, "w") as f:
        json.dump({k: v for k, v in results_by_show.items()}, f, indent=2)
    print(f"\n→ {out_json}")

    plot_comparison(results_by_show)
    print_examples(results_by_show)


if __name__ == "__main__":
    main()
