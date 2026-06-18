import html
import json
import os
import random
import re
import unicodedata
import warnings
from collections import deque
from multiprocessing import Pool
from pathlib import Path
from typing import Deque, List, Optional, Union

import torch
import torchaudio
from tqdm import tqdm
from whisper.audio import load_audio
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from whisper.utils import format_timestamp
from whisper_prep.audio.vad import silero_speech_ratio
from whisper_prep.generation.typing import PromptNode, Record, Utterance
import csv
from collections import defaultdict

DURATION = 30000  # 30 seconds in milliseconds
SAMPLE_RATE = 16000
DURATION_IN_SAMPLES = int(DURATION * SAMPLE_RATE / 1000)
_WORKER_PROCESSOR = None
_WORKER_OUTPUT = None


def _init_transcripts_worker(processor_kwargs: dict, output_dir: str) -> None:
    global _WORKER_PROCESSOR, _WORKER_OUTPUT
    torch.set_num_threads(1)
    worker_output = Path(output_dir, f"data.worker.{os.getpid()}.ljson")
    worker_output.unlink(missing_ok=True)
    worker_kwargs = dict(processor_kwargs)
    worker_kwargs["output"] = str(worker_output)
    _WORKER_OUTPUT = worker_output
    _WORKER_PROCESSOR = DataProcessor(**worker_kwargs)


def _process_transcripts_tsv_row(row: dict) -> dict:
    if _WORKER_PROCESSOR is None or _WORKER_OUTPUT is None:
        raise RuntimeError("Transcript worker was not initialized")

    before_size = _WORKER_OUTPUT.stat().st_size if _WORKER_OUTPUT.exists() else 0
    filtered_records, error = _WORKER_PROCESSOR._process_transcripts_tsv_row(row)
    records_written = 0
    if _WORKER_OUTPUT.exists():
        with _WORKER_OUTPUT.open(encoding="utf-8") as f:
            f.seek(before_size)
            records_written = sum(1 for _ in f)

    return {
        "records_written": records_written,
        "filtered_records": filtered_records,
        "error": error,
    }


def _process_audio_path_worker(audio_path_str: str) -> dict:
    """Pool worker for the folder-based (SRT-on-disk) timestamping path."""
    if _WORKER_PROCESSOR is None or _WORKER_OUTPUT is None:
        raise RuntimeError("Folder worker was not initialized")
    return _WORKER_PROCESSOR._process_one_audio(Path(audio_path_str))


class DataProcessor:
    def __init__(
        self,
        audio_dir: str,
        transcript_dir: str,
        with_timestamps: bool = True,
        data_file: Optional[str] = None,
        transcript_formats: List[str] = [
            "{id}.srt",
            "{id}.vtt",
        ],
        language: str = "de",
        output: str = "data.json",
        dump_dir: str = "dump",
        timestamp_resolution: int = 20,
        max_prompt_length: int = 223,  # 223 tokens and some extra for the time stamps.
        max_tokens_length: int = 219,
        subsampling_factor_for_silence: int = 1,
        keep_empty_chance: float = 0.0,
        rep_threshold: int = 3,
        tokenizer_type: str = "multilingual",
        normalize_unicode: bool = False,
        cut_initial_audio: bool = False,
        filter_segment_words: Optional[List[str]] = None,
        drop_text: Optional[List[str]] = None,
        transcripts_tsv: Optional[str] = None,
        validate_empty_with_vad: bool = False,
        empty_vad_max_speech_ratio: float = 0.06,
        n_jobs: int = 1,
    ) -> None:
        self.with_timestamps = with_timestamps
        self.audio_dir = audio_dir
        self.transcript_dir = transcript_dir
        self.data_file = data_file
        self.transcript_formats = transcript_formats
        self.language = language
        self.output = output
        self.dump_dir = dump_dir
        self.timestamp_resolution = timestamp_resolution
        self.max_prompt_length = max_prompt_length
        self.max_tokens_length = max_tokens_length
        self.subsampling_factor_for_silence = subsampling_factor_for_silence
        self.keep_empty_chance = keep_empty_chance
        self.rep_threshold = rep_threshold
        self.tokenizer_type = tokenizer_type
        self.normalize_unicode = normalize_unicode
        self.cut_initial_audio = cut_initial_audio
        self.filter_segment_words = filter_segment_words
        self.drop_text = drop_text
        self.transcripts_tsv = transcripts_tsv
        self.validate_empty_with_vad = validate_empty_with_vad
        self.empty_vad_max_speech_ratio = empty_vad_max_speech_ratio
        self.n_jobs = n_jobs
        self.filtered_segment_records: List[dict] = []

        self._verify_args()

        self.tokenizer = get_tokenizer(
            multilingual=(self.tokenizer_type == "multilingual")
        )
        Path(self.dump_dir).mkdir(parents=True, exist_ok=True)

    def _verify_args(self) -> None:
        if self.with_timestamps:
            if not self.transcripts_tsv:
                if not self.audio_dir or not self.transcript_dir:
                    raise ValueError(
                        "`audio_dir` and `transcript_dir` must be set when `with_timestamps` is True and no transcripts_tsv provided"
                    )
            if self.timestamp_resolution % 20 != 0:
                raise ValueError(
                    "`timestamps_resolution` must be multiples of 20ms. "
                    f"Got {self.timestamp_resolution}"
                )
        else:
            if not self.data_file:
                raise ValueError(
                    "`data_file` must be set when `with_timestamps` is False"
                )

        if self.language not in LANGUAGES:
            if self.language in TO_LANGUAGE_CODE:
                self.language = TO_LANGUAGE_CODE[self.language]
            else:
                raise ValueError(f"Unsupported language: {self.language}")

        if self.tokenizer_type not in ["multilingual", "english"]:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

        if Path(self.output).exists():
            raise ValueError(f"Output file {self.output} already exists")

        if not 0 <= self.keep_empty_chance <= 1:
            raise ValueError(
                f"keep_empty_chance must be between 0 and 1, got {self.keep_empty_chance}"
            )

        if not 0 <= self.empty_vad_max_speech_ratio <= 1:
            raise ValueError(
                "empty_vad_max_speech_ratio must be between 0 and 1, "
                f"got {self.empty_vad_max_speech_ratio}"
            )

        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be at least 1, got {self.n_jobs}")

    def run(self) -> None:
        if self.with_timestamps:
            self._process_with_timestamps()
        else:
            self._process_without_timestamps()

        self._write_filtered_segments()

        if self.subsampling_factor_for_silence > 1:
            self._subsample_silence()

    def _process_without_timestamps(self) -> None:
        records = []
        with open(self.data_file, encoding="utf-8") as f:
            for line in f:
                audio_path, text = line.strip().split("\t")
                if self.normalize_unicode:
                    text = unicodedata.normalize("NFKC", text)

                tokens = self.tokenizer.encode(text)
                if len(tokens) > self.max_tokens_length:
                    print(
                        f"Skipping {audio_path} ({text}) because it is too long "
                        f"({len(tokens)} tokens)"
                    )
                    continue

                record = Record(
                    audio_path=audio_path, text=text, language=self.language
                )
                records.append(record)

        self.write_records(records, self.output)

    def _drop_repeated_utterances(
        self, utterances: List[Utterance], threshold: int = 3
    ) -> List[Utterance]:
        kept, _ = self._drop_repeated_utterances_with_records(utterances, threshold)
        return kept

    def _drop_repeated_utterances_with_records(
        self, utterances: List[Utterance], threshold: int = 3
    ) -> tuple[List[Utterance], List[dict]]:
        if not utterances:
            return [], []

        # This function assumes utterances are already sorted by start time.
        result_utterances = []
        dropped_records = []
        repeat_count = 1
        last_text = None

        def flush_group(end_index: int) -> None:
            group = utterances[end_index - repeat_count : end_index]
            if repeat_count < threshold:
                result_utterances.extend(group)
                return

            dropped_records.append(
                {
                    "start_ms": min(utterance.start for utterance in group),
                    "end_ms": max(utterance.end for utterance in group),
                    "text": group[0].text,
                    "matched_word": "repeated_hallucination",
                }
            )

        for i in range(len(utterances)):
            current_text = utterances[i].text

            if i == 0:
                last_text = current_text
                continue

            if current_text == last_text:
                repeat_count += 1
            else:
                flush_group(i)
                # Reset count and update last_text
                repeat_count = 1
                last_text = current_text

        # Check the last sequence at the end of the list
        flush_group(len(utterances))

        return result_utterances, dropped_records

    def _drop_single_letter_utterance(
        self, utterances: List[Utterance]
    ) -> List[Utterance]:
        sanitized_utterances = []
        for utterance in utterances:
            if len(utterance.text.strip()) > 0:
                sanitized_utterances.append(utterance)

        return sanitized_utterances

    @staticmethod
    def _is_punctuation_hallucination(text: str) -> bool:
        compact = re.sub(r"\s+", "", text or "")
        if len(compact) < 3:
            return False
        has_word_character = re.search(r"\w", compact, flags=re.UNICODE)
        return has_word_character is None

    def _drop_punctuation_hallucinations_with_records(
        self, utterances: List[Utterance]
    ) -> tuple[List[Utterance], List[dict]]:
        kept = []
        dropped_records = []
        for utterance in utterances:
            if self._is_punctuation_hallucination(utterance.text):
                dropped_records.append(
                    {
                        "start_ms": utterance.start,
                        "end_ms": utterance.end,
                        "text": utterance.text,
                        "matched_word": "punctuation_hallucination",
                    }
                )
            else:
                kept.append(utterance)
        return kept, dropped_records

    def _sanitize_utterances(
        self,
        utterances: List[Utterance],
        filtered_out: Optional[List[dict]] = None,
        source_id: Optional[str] = None,
        transcript_path: Optional[Union[str, Path]] = None,
    ) -> List[Utterance]:
        if not utterances:
            return []

        # Remove duplicate hallucinations
        utterances, dropped_repeats = self._drop_repeated_utterances_with_records(
            utterances
        )
        if filtered_out is not None:
            for record in dropped_repeats:
                filtered_out.append(
                    {
                        "speech_id": source_id
                        or (Path(transcript_path).stem if transcript_path else ""),
                        "transcript_path": str(transcript_path or ""),
                        **record,
                    }
                )

        utterances, dropped_punctuation = (
            self._drop_punctuation_hallucinations_with_records(utterances)
        )
        if filtered_out is not None:
            for record in dropped_punctuation:
                filtered_out.append(
                    {
                        "speech_id": source_id
                        or (Path(transcript_path).stem if transcript_path else ""),
                        "transcript_path": str(transcript_path or ""),
                        **record,
                    }
                )

        # Drop single letter predictions
        utterances = self._drop_single_letter_utterance(utterances)

        # Add dummy utterances for easier logic
        utterances.append(Utterance(text=None, start=9999999999999, end=99999999999999))
        utterances.insert(0, Utterance(text=None, start=-100, end=-99))

        # Sort utterances by start time first
        utterances.sort(key=lambda u: u.start if u.start is not None else 0)

        sanitized_utterances = []

        for i, _ in enumerate(utterances):
            current = utterances[i]
            # Handle cases where the start time is invalid with respect to the end time
            if current.start >= current.end:
                # Check if it should be added to the one before or after
                previous_range = range(
                    utterances[i - 1].start, utterances[i - 1].end + 1
                )
                if current.start in previous_range or current.end in previous_range:
                    # Add to the previous.
                    sanitized_utterances[-1] = Utterance(
                        text=sanitized_utterances[-1].text + " " + current.text,
                        start=min(sanitized_utterances[-1].start, current.start),
                        end=max(sanitized_utterances[-1].end, current.end),
                    )
                else:
                    # If this utterance is after the previous, add it to the following one.
                    utterances[i + 1] = Utterance(
                        text=current.text + " " + utterances[i + 1].text,
                        start=min(utterances[i + 1].start, current.start),
                        end=max(utterances[i + 1].end, current.end),
                    )
            else:
                sanitized_utterances.append(current)
            # Update end before.
        sanitized_utterances.pop()
        sanitized_utterances.pop(0)

        # Remove duplicate hallucinations, which happen often when transcribi
        return sanitized_utterances

    def _process_with_timestamps(self) -> None:
        if self.transcripts_tsv:
            if self.n_jobs > 1:
                self._process_transcripts_tsv_parallel()
                return
            self._process_transcripts_tsv_sequential()
            return
        audio_paths = list(Path(self.audio_dir).iterdir())

        # Cutting each generated audio + its SRT into <=30s records is fully
        # independent per file, so parallelize it when n_jobs > 1.
        if self.n_jobs > 1 and len(audio_paths) > 1:
            self._process_folder_parallel(audio_paths)
            return

        for audio_path in tqdm(audio_paths):
            result = self._process_one_audio(audio_path)
            self.filtered_segment_records.extend(result["filtered_records"])
            if result["error"]:
                print(result["error"])

    def _process_one_audio(self, audio_path: Path) -> dict:
        """Process a single generated audio file and its transcript.

        Reads the matching SRT/VTT, creates timestamped records and appends
        them to ``self.output``. Returns the filtered-segment records plus an
        optional error string (instead of raising) so it is safe to call from
        a multiprocessing pool worker.
        """
        speech_id = audio_path.stem

        for fmt in self.transcript_formats:
            transcript_path = Path(self.transcript_dir) / fmt.format(id=speech_id)
            if not transcript_path.exists():
                continue
            try:
                filtered_for_speech: List[dict] = []
                if transcript_path.suffix == ".srt":
                    utterances_for_speech = self.read_utterances_from_srt(
                        transcript_path,
                        self.normalize_unicode,
                        self.filter_segment_words,
                        self.drop_text,
                        filtered_for_speech,
                        speech_id,
                    )
                elif transcript_path.suffix == ".vtt":
                    utterances_for_speech = self.read_utterances_from_vtt(
                        transcript_path,
                        self.normalize_unicode,
                        self.filter_segment_words,
                        self.drop_text,
                        filtered_for_speech,
                        speech_id,
                    )
                else:
                    continue
                # Sanitize utterances, if necessary. Takes care of some random
                # timestamp errors produced by the VAD of whisperx.
                if not self._is_valid_utterances(utterances_for_speech, 0):
                    utterances_for_speech = self._sanitize_utterances(
                        utterances_for_speech,
                        filtered_for_speech,
                        speech_id,
                        transcript_path,
                    )
                blocked_intervals = [
                    (r["start_ms"], r["end_ms"]) for r in filtered_for_speech
                ]
                records = self._create_records_with_timestamps(
                    utterances_for_speech,
                    audio_path,
                    speech_id,
                    blocked_intervals=blocked_intervals,
                )
                self.write_records(records, self.output)
                return {"filtered_records": filtered_for_speech, "error": None}
            except Exception as e:
                return {
                    "filtered_records": [],
                    "error": (
                        f"Skipping {transcript_path} due to an error in the "
                        f"transcript: {e}"
                    ),
                }

        return {
            "filtered_records": [],
            "error": f"Transcript file not found for {speech_id}",
        }

    def _process_folder_parallel(self, audio_paths: List[Path]) -> None:
        """Parallel version of the folder-based timestamping path.

        Mirrors :meth:`_process_transcripts_tsv_parallel`: each worker owns its
        own output shard, then the shards are concatenated into ``self.output``.
        """
        parts_dir = Path(self.output).parent / "_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        for part_path in parts_dir.glob("data.worker.*.ljson"):
            part_path.unlink()

        processor_kwargs = {
            "audio_dir": self.audio_dir,
            "transcript_dir": self.transcript_dir,
            "with_timestamps": self.with_timestamps,
            "data_file": self.data_file,
            "transcript_formats": self.transcript_formats,
            "language": self.language,
            "dump_dir": self.dump_dir,
            "timestamp_resolution": self.timestamp_resolution,
            "max_prompt_length": self.max_prompt_length,
            "max_tokens_length": self.max_tokens_length,
            "subsampling_factor_for_silence": 1,
            "keep_empty_chance": self.keep_empty_chance,
            "rep_threshold": self.rep_threshold,
            "tokenizer_type": self.tokenizer_type,
            "normalize_unicode": self.normalize_unicode,
            "cut_initial_audio": self.cut_initial_audio,
            "filter_segment_words": self.filter_segment_words,
            "drop_text": self.drop_text,
            "transcripts_tsv": None,
            "validate_empty_with_vad": self.validate_empty_with_vad,
            "empty_vad_max_speech_ratio": self.empty_vad_max_speech_ratio,
            "n_jobs": 1,
        }
        worker_count = min(self.n_jobs, len(audio_paths))
        print(f"Cutting {len(audio_paths)} SRTs into segments with {worker_count} workers")

        audio_path_strs = [str(p) for p in audio_paths]
        with Pool(
            processes=worker_count,
            initializer=_init_transcripts_worker,
            initargs=(processor_kwargs, str(parts_dir)),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_audio_path_worker, audio_path_strs),
                total=len(audio_path_strs),
                desc="Cutting SRTs into segments",
            ):
                self.filtered_segment_records.extend(result["filtered_records"])
                if result["error"]:
                    print(result["error"])

        Path(self.output).unlink(missing_ok=True)
        with open(self.output, "w", encoding="utf-8") as outfile:
            for part_path in sorted(parts_dir.glob("data.worker.*.ljson")):
                with part_path.open(encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)

    def _process_transcripts_tsv_sequential(self) -> None:
        if self.transcripts_tsv:
            with open(self.transcripts_tsv, encoding="utf-8") as tsvfile:
                # Pre-count rows so tqdm can display ETA and progress percentage.
                total_rows = max(0, sum(1 for _ in tsvfile) - 1)
                tsvfile.seek(0)
                reader = csv.DictReader(tsvfile, delimiter="\t")
                for row in tqdm(
                    reader,
                    total=total_rows,
                    desc="Processing TSV transcripts",
                ):
                    srt_path = Path(row["srt_path"])
                    audio_path = Path(row["audio_path"])
                    speech_id = row.get("id") or audio_path.stem
                    orig_lang = self.language
                    self.language = row.get("language") or self.language
                    filtered_for_speech: List[dict] = []
                    try:
                        if srt_path.suffix == ".srt":
                            utterances = self.read_utterances_from_srt(
                                srt_path,
                                self.normalize_unicode,
                                self.filter_segment_words,
                                self.drop_text,
                                filtered_for_speech,
                                speech_id,
                            )
                        elif srt_path.suffix == ".vtt":
                            utterances = self.read_utterances_from_vtt(
                                srt_path,
                                self.normalize_unicode,
                                self.filter_segment_words,
                                self.drop_text,
                                filtered_for_speech,
                                speech_id,
                            )
                        else:
                            raise ValueError(
                                f"Unsupported transcript format: {srt_path.suffix}"
                            )
                        if not self._is_valid_utterances(utterances, 0):
                            utterances = self._sanitize_utterances(
                                utterances,
                                filtered_for_speech,
                                speech_id,
                                srt_path,
                            )
                        blocked_intervals = [
                            (r["start_ms"], r["end_ms"]) for r in filtered_for_speech
                        ]
                        self.filtered_segment_records.extend(filtered_for_speech)
                        records = self._create_records_with_timestamps(
                            utterances,
                            audio_path,
                            speech_id,
                            blocked_intervals=blocked_intervals,
                        )
                        self.write_records(records, self.output)
                    except Exception as e:
                        print(e)
                        print(f"Skipping {srt_path} due to an error in the transcript")
                    finally:
                        self.language = orig_lang
            return

    def _process_transcripts_tsv_parallel(self) -> None:
        if not self.transcripts_tsv:
            return

        with open(self.transcripts_tsv, encoding="utf-8") as tsvfile:
            rows = list(csv.DictReader(tsvfile, delimiter="\t"))

        parts_dir = Path(self.output).parent / "_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        for part_path in parts_dir.glob("data.worker.*.ljson"):
            part_path.unlink()

        processor_kwargs = {
            "audio_dir": self.audio_dir,
            "transcript_dir": self.transcript_dir,
            "with_timestamps": self.with_timestamps,
            "data_file": self.data_file,
            "transcript_formats": self.transcript_formats,
            "language": self.language,
            "dump_dir": self.dump_dir,
            "timestamp_resolution": self.timestamp_resolution,
            "max_prompt_length": self.max_prompt_length,
            "max_tokens_length": self.max_tokens_length,
            "subsampling_factor_for_silence": 1,
            "keep_empty_chance": self.keep_empty_chance,
            "rep_threshold": self.rep_threshold,
            "tokenizer_type": self.tokenizer_type,
            "normalize_unicode": self.normalize_unicode,
            "cut_initial_audio": self.cut_initial_audio,
            "filter_segment_words": self.filter_segment_words,
            "drop_text": self.drop_text,
            "transcripts_tsv": self.transcripts_tsv,
            "validate_empty_with_vad": self.validate_empty_with_vad,
            "empty_vad_max_speech_ratio": self.empty_vad_max_speech_ratio,
            "n_jobs": 1,
        }
        worker_count = min(self.n_jobs, len(rows))
        print(f"Processing TSV transcripts with {worker_count} workers")

        with Pool(
            processes=worker_count,
            initializer=_init_transcripts_worker,
            initargs=(processor_kwargs, str(parts_dir)),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_transcripts_tsv_row, rows),
                total=len(rows),
                desc="Processing TSV transcripts",
            ):
                self.filtered_segment_records.extend(result["filtered_records"])
                if result["error"]:
                    print(result["error"])

        Path(self.output).unlink(missing_ok=True)
        with open(self.output, "w", encoding="utf-8") as outfile:
            for part_path in sorted(parts_dir.glob("data.worker.*.ljson")):
                with part_path.open(encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)

    def _process_transcripts_tsv_row(self, row: dict) -> tuple[List[dict], Optional[str]]:
        srt_path = Path(row["srt_path"])
        audio_path = Path(row["audio_path"])
        speech_id = row.get("id") or audio_path.stem
        orig_lang = self.language
        self.language = row.get("language") or self.language
        filtered_for_speech: List[dict] = []
        try:
            if srt_path.suffix == ".srt":
                utterances = self.read_utterances_from_srt(
                    srt_path,
                    self.normalize_unicode,
                    self.filter_segment_words,
                    self.drop_text,
                    filtered_for_speech,
                    speech_id,
                )
            elif srt_path.suffix == ".vtt":
                utterances = self.read_utterances_from_vtt(
                    srt_path,
                    self.normalize_unicode,
                    self.filter_segment_words,
                    self.drop_text,
                    filtered_for_speech,
                    speech_id,
                )
            else:
                raise ValueError(f"Unsupported transcript format: {srt_path.suffix}")
            if not self._is_valid_utterances(utterances, 0):
                utterances = self._sanitize_utterances(
                    utterances,
                    filtered_for_speech,
                    speech_id,
                    srt_path,
                )
            blocked_intervals = [
                (r["start_ms"], r["end_ms"]) for r in filtered_for_speech
            ]
            records = self._create_records_with_timestamps(
                utterances,
                audio_path,
                speech_id,
                blocked_intervals=blocked_intervals,
            )
            self.write_records(records, self.output)
            return filtered_for_speech, None
        except Exception as e:
            return (
                filtered_for_speech,
                f"{e}\nSkipping {srt_path} due to an error in the transcript",
            )
        finally:
            self.language = orig_lang

    @staticmethod
    def read_utterances_from_srt(
        transcript_path: Union[str, Path],
        normalize_unicode: bool = False,
        filter_segment_words: Optional[List[str]] = None,
        drop_text: Optional[List[str]] = None,
        filtered_out: Optional[List[dict]] = None,
        source_id: Optional[str] = None,
    ) -> List[Utterance]:
        utterances = []
        with open(transcript_path, encoding="utf-8") as f:
            lines = f.readlines()
            timestamps_indices = [i for i, line in enumerate(lines) if " --> " in line]
            timestamps_indices.append(
                len(lines) + 1
            )  # a dummy index to make the loop below simple

            for i in range(len(timestamps_indices) - 1):
                utterance_start = timestamps_indices[i]
                next_utterance_start = timestamps_indices[i + 1]

                start_time, end_time = DataProcessor._parse_timestamp_line(
                    lines[utterance_start]
                )

                # `next_utterance_start - 1` corresponds to an index number of the utterance and
                # `next_utterance_start - 2` corresponds to a newline character, thus the text is
                # included between [`utterance_start + 1`, `next_utterance_start - 2`).
                text = " ".join(
                    [
                        line.strip()
                        for line in lines[
                            utterance_start + 1 : next_utterance_start - 2
                        ]
                    ]
                ).strip()
                text = DataProcessor._clean_subtitle_markup(text)
                if normalize_unicode:
                    text = unicodedata.normalize("NFKC", text)
                text = DataProcessor._drop_text_fragments(text, drop_text)
                if not text:
                    continue
                # Skip if single character
                if len(text) == 1:
                    continue
                # Filter out utterances containing specific words, if specified
                contains_filter_words = False
                matched_word = None
                if filter_segment_words is not None:
                    for word in filter_segment_words:
                        if word.lower() in text.lower():
                            contains_filter_words = True
                            matched_word = word
                            break

                if contains_filter_words:
                    if filtered_out is not None:
                        filtered_out.append(
                            {
                                "speech_id": source_id
                                or Path(transcript_path).stem,
                                "transcript_path": str(transcript_path),
                                "start_ms": start_time,
                                "end_ms": end_time,
                                "text": text,
                                "matched_word": matched_word or "",
                            }
                        )
                    continue

                utterances.append(Utterance(text=text, start=start_time, end=end_time))

        return utterances

    @staticmethod
    def read_utterances_from_vtt(
        transcript_path: Union[str, Path],
        normalize_unicode: bool = False,
        filter_segment_words: Optional[List[str]] = None,
        drop_text: Optional[List[str]] = None,
        filtered_out: Optional[List[dict]] = None,
        source_id: Optional[str] = None,
    ) -> List[Utterance]:
        utterances = []
        with open(transcript_path, encoding="utf-8") as f:
            lines = f.readlines()
            timestamps_indices = [i for i, line in enumerate(lines) if " --> " in line]
            timestamps_indices.append(
                len(lines) + 1
            )  # a dummy index to make the loop below simple

            for i in range(len(timestamps_indices) - 1):
                utterance_start = timestamps_indices[i]
                next_utterance_start = timestamps_indices[i + 1]

                start_time, end_time = DataProcessor._parse_timestamp_line(
                    lines[utterance_start]
                )

                # `next_utterance_start - 1` corresponds to a newline, thus the text is included
                # between [`utterance_start + 1`, `next_utterance_start - 1`).
                text = " ".join(
                    [
                        line.strip()
                        for line in lines[
                            utterance_start + 1 : next_utterance_start - 1
                        ]
                    ]
                ).strip()
                text = DataProcessor._clean_subtitle_markup(text)
                if normalize_unicode:
                    text = unicodedata.normalize("NFKC", text)
                text = DataProcessor._drop_text_fragments(text, drop_text)
                # Filter out empty utterances
                if not text:
                    continue
                # Skip if single dot
                if text == ".":
                    continue
                # Filter out utterances containing specific words, if specified
                if filter_segment_words:
                    matched_word = None
                    for word in filter_segment_words:
                        if word.lower() in text.lower():
                            matched_word = word
                            break
                    if matched_word is not None:
                        if filtered_out is not None:
                            filtered_out.append(
                                {
                                    "speech_id": source_id
                                    or Path(transcript_path).stem,
                                    "transcript_path": str(transcript_path),
                                    "start_ms": start_time,
                                    "end_ms": end_time,
                                    "text": text,
                                    "matched_word": matched_word,
                                }
                            )
                        continue

                utterances.append(Utterance(text=text, start=start_time, end=end_time))

        return utterances

    @staticmethod
    def _clean_subtitle_markup(text: str) -> str:
        text = html.unescape(text)
        text = re.sub(r"</?[^>]+>", "", text)
        text = re.sub(r"\s+<[^>]*$", "", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    @staticmethod
    def _drop_text_fragments(text: str, drop_text: Optional[List[str]] = None) -> str:
        if not drop_text:
            return text

        for fragment in drop_text:
            if not fragment:
                continue
            text = re.sub(re.escape(fragment), "", text, flags=re.IGNORECASE)

        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    @staticmethod
    def _parse_timestamp_line(line: str) -> tuple[int, int]:
        start_time, end_time = line.strip().split(" --> ", 1)
        end_time = end_time.split()[0]
        return (
            DataProcessor.str_to_milliseconds(start_time),
            DataProcessor.str_to_milliseconds(end_time),
        )

    def _write_filtered_segments(self) -> None:
        if not self.filtered_segment_records:
            return

        out_folder = Path(self.output).parent
        if out_folder.name == "created_dataset":
            out_folder = out_folder.parent

        grouped = defaultdict(list)
        for record in self.filtered_segment_records:
            grouped[record.get("matched_word", "")].append(record)

        for word, records in grouped.items():
            if not records:
                continue
            out_path = out_folder / f"filtered_{word}_examples.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "speech_id",
                        "transcript_path",
                        "start_ms",
                        "end_ms",
                        "text",
                        "matched_word",
                    ],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(records)

    def _create_records_with_timestamps(
        self,
        utterances: List[Utterance],
        audio_path: Path,
        speech_id: Optional[str] = None,
        blocked_intervals: Optional[List[tuple]] = None,
    ) -> List[Record]:
        audio = torch.tensor(load_audio(audio_path))
        dump_dir = Path(self.dump_dir) / (speech_id if speech_id else audio_path.stem)
        dump_dir.mkdir(parents=True, exist_ok=True)
        audio_duration_ms = int(audio.size(0) * 1000 / SAMPLE_RATE)
        safe_spans = self._get_safe_spans(audio_duration_ms, blocked_intervals or [])
        records = []
        utterances = sorted(utterances, key=lambda u: u.start)
        span_cursor = 0
        for i, (span_start, span_end) in enumerate(safe_spans):
            if span_start >= span_end:
                continue
            # Restart aggregation across filtered-word gaps.
            prompt_buffer: Deque[PromptNode] = deque()

            span_utterances = []
            while span_cursor < len(utterances):
                utterance = utterances[span_cursor]
                if utterance.end is None or utterance.start is None:
                    span_cursor += 1
                    continue
                if utterance.end <= span_start:
                    span_cursor += 1
                    continue
                if utterance.start >= span_end:
                    break
                # Defensive: if a subtitle intersects a blocked interval boundary, drop it.
                if utterance.start < span_start or utterance.end > span_end:
                    span_cursor += 1
                    continue
                span_utterances.append(utterance)
                span_cursor += 1

            if not span_utterances:
                records.extend(
                    self._create_empty_records_for_span(
                        audio, audio_path, dump_dir, span_start, span_end
                    )
                )
                continue

            # Optionally trim initial audio only for the first safe span.
            if i == 0 and self.cut_initial_audio:
                segment_start = max(span_start, span_utterances[0].start - 1000)
            else:
                segment_start = span_start

            idx = 0
            while idx < len(span_utterances):
                segment_end = min(segment_start + DURATION, span_end)
                if segment_start >= segment_end:
                    break

                # If the utterance is included in the segment and longer than the segment, skip it.
                if (
                    span_utterances[idx].start < segment_end
                    and span_utterances[idx].start + DURATION < span_utterances[idx].end
                ):
                    segment_start = span_utterances[idx].end
                    idx += 1
                    continue

                prompt = self._get_prompt(prompt_buffer)

                segment_utterances = []
                next_segment_start = None
                while (
                    idx < len(span_utterances)
                    and span_utterances[idx].start < segment_end
                ):
                    if span_utterances[idx].end > segment_end:
                        next_segment_start = span_utterances[idx].start
                        break
                    segment_utterances.append(span_utterances[idx])
                    idx += 1

                if not self._is_valid_utterances(segment_utterances, segment_start):
                    tqdm.write(
                        f"Skipping {audio_path} ({format_timestamp(segment_start / 1000)}-"
                        f"{format_timestamp(segment_end / 1000)}) because it contains invalid "
                        f"utterances: {segment_utterances}"
                    )
                    prompt_buffer.clear()
                    segment_start = max(segment_end, segment_utterances[-1].end)
                    continue

                tokens_length = 0
                segment_text = []
                for utterance in segment_utterances:
                    start_token = self._get_time_token(
                        utterance.start, segment_start, audio_path
                    )
                    end_token = self._get_time_token(
                        utterance.end, segment_start, audio_path
                    )
                    utterance_text = self._add_leading_space(utterance.text)
                    segment_text.extend([start_token, utterance_text, end_token])
                    new_prompt_length = len(self.tokenizer.encode(utterance_text)) + 2
                    new_prompt_node = PromptNode(
                        start_token + utterance_text + end_token, new_prompt_length
                    )
                    tokens_length += new_prompt_length

                    prompt_buffer.append(new_prompt_node)

                if tokens_length > self.max_tokens_length:
                    tqdm.write(
                        f"Skipping {audio_path} ({format_timestamp(segment_start / 1000)}-"
                        f"{format_timestamp(segment_end / 1000)}) because it is too long "
                        f"({tokens_length} tokens)"
                    )
                elif not segment_utterances and random.random() >= self.keep_empty_chance:
                    pass
                else:
                    audio_segment_end = self._get_audio_segment_end(
                        segment_utterances,
                        segment_start,
                        segment_end,
                        next_segment_start,
                    )
                    segment_audio_path = self._save_segment_audio(
                        audio, segment_start, audio_segment_end, dump_dir
                    )
                    empty_segment_failed_vad = (
                        not segment_utterances
                        and not self._empty_segment_passes_vad(segment_audio_path)
                    )
                    if not empty_segment_failed_vad:
                        record = Record(
                            audio_path=segment_audio_path,
                            language=self.language,
                            text="".join(segment_text),
                            prompt=prompt,
                        )
                        records.append(record)

                if next_segment_start is not None:
                    segment_start = next_segment_start
                elif len(segment_utterances) == 0:
                    segment_start += DURATION
                else:
                    segment_start = segment_utterances[-1].end

            if segment_start < span_end:
                records.extend(
                    self._create_empty_records_for_span(
                        audio, audio_path, dump_dir, segment_start, span_end
                    )
                )

        return records

    def _create_empty_records_for_span(
        self,
        audio: torch.Tensor,
        audio_path: Path,
        dump_dir: Path,
        span_start: int,
        span_end: int,
    ) -> List[Record]:
        records = []
        segment_start = span_start
        while segment_start < span_end:
            segment_end = min(segment_start + DURATION, span_end)
            if segment_start >= segment_end:
                break
            if random.random() < self.keep_empty_chance:
                segment_audio_path = self._save_segment_audio(
                    audio, segment_start, segment_end, dump_dir
                )
                if not self._empty_segment_passes_vad(segment_audio_path):
                    segment_start = segment_end
                    continue
                records.append(
                    Record(
                        audio_path=segment_audio_path,
                        language=self.language,
                        text="",
                        prompt="",
                    )
                )
            segment_start = segment_end
        return records

    def _empty_segment_passes_vad(self, segment_audio_path: str) -> bool:
        if not self.validate_empty_with_vad:
            return True

        speech_ratio = silero_speech_ratio(segment_audio_path)
        if speech_ratio <= self.empty_vad_max_speech_ratio:
            return True

        tqdm.write(
            f"Skipping empty-text segment {segment_audio_path} because VAD found "
            f"{speech_ratio:.2%} speech"
        )
        Path(segment_audio_path).unlink(missing_ok=True)
        return False

    def _save_segment_audio(
        self, audio: torch.Tensor, segment_start: int, segment_end: int, dump_dir: Path
    ) -> str:
        audio_start_idx = int(segment_start * SAMPLE_RATE / 1000)
        audio_end_idx = int(segment_end * SAMPLE_RATE / 1000)
        segment_audio_path = str((dump_dir / f"{segment_start}.mp3").absolute())
        segment_audio = audio[
            audio_start_idx : min(audio_end_idx, audio_start_idx + DURATION_IN_SAMPLES, audio.size(0))
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The 'encoding' parameter is not fully supported by TorchCodec AudioEncoder.",
                category=UserWarning,
            )
            torchaudio.save(
                segment_audio_path,
                segment_audio.unsqueeze(0),
                SAMPLE_RATE,
                encoding="mp3",
            )
        return segment_audio_path

    def _get_audio_segment_end(
        self,
        segment_utterances: List[Utterance],
        segment_start: int,
        segment_end: int,
        next_segment_start: Optional[int],
    ) -> int:
        if next_segment_start is None:
            return segment_end
        if not segment_utterances:
            return min(next_segment_start, segment_end)

        rounded_utterance_end = max(
            segment_start
            + round((utterance.end - segment_start) / self.timestamp_resolution)
            * self.timestamp_resolution
            for utterance in segment_utterances
        )
        return min(max(next_segment_start, rounded_utterance_end), segment_end)

    @staticmethod
    def _merge_intervals(intervals: List[tuple]) -> List[tuple]:
        if not intervals:
            return []
        merged = []
        for start, end in sorted(intervals):
            if start is None or end is None:
                continue
            if end <= start:
                continue
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return [(start, end) for start, end in merged]

    def _get_safe_spans(
        self, audio_duration_ms: int, blocked_intervals: List[tuple]
    ) -> List[tuple]:
        blocked = self._merge_intervals(blocked_intervals)
        if not blocked:
            return [(0, audio_duration_ms)]

        safe_spans = []
        cursor = 0
        for start, end in blocked:
            start = max(0, min(start, audio_duration_ms))
            end = max(0, min(end, audio_duration_ms))
            if cursor < start:
                safe_spans.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < audio_duration_ms:
            safe_spans.append((cursor, audio_duration_ms))
        return safe_spans

    def _is_valid_utterances(
        self, utterances: List[Utterance], segment_start: int
    ) -> bool:
        if len(utterances) == 0:
            return True

        for utterance in utterances:
            # Check the utterances' start times are in the segment
            if utterance.start < segment_start:
                return False
            if utterance.start > utterance.end:
                return False
            if self._is_punctuation_hallucination(utterance.text):
                return False

        # Check the utterances do not overlap
        for i in range(len(utterances) - 1):
            if utterances[i].end > utterances[i + 1].start:
                return False

        # Check for repeated words three or more times consecutively
        last_text = utterances[0].text
        repeat_count = 1
        for i in range(1, len(utterances)):
            if utterances[i].text == last_text:
                repeat_count += 1
                if repeat_count >= self.rep_threshold:
                    return False
            else:
                last_text = utterances[i].text
                repeat_count = 1

        return True

    def _add_leading_space(self, text: str) -> str:
        """
        Add a leading space to the text if the language uses spaces to separate words.
        For languages that do not use spaces, namely Chinese, Japanese, Thai, Lao, and
        Burmese, return the text as is.
        """
        if self.language in ["zh", "ja", "th", "lo", "my"]:
            return text
        else:
            return " " + text

    @staticmethod
    def str_to_milliseconds(s: str) -> int:
        """
        Convert a string in the format of "00:00:00,000" to milliseconds.
        """
        if "," in s:
            time, miliseconds = s.split(",")
        elif "." in s:
            time, miliseconds = s.split(".")
        else:
            raise ValueError(
                f"Invalid time format: {s}. Must be in the format of 00:00:00,000 or 00:00:00.000"
            )
        hours, minutes, seconds = time.split(":")
        hours = int(hours)
        minutes = int(minutes)
        seconds = int(seconds)
        miliseconds = int(miliseconds)
        return (hours * 3600 + minutes * 60 + seconds) * 1000 + miliseconds

    def _get_time_token(self, time: int, segment_start: int, audio_path: Path) -> str:
        """
        Get the time token for the given time.

        Args:
            time: Time in milliseconds
            segment_start: Start time of the segment in milliseconds

        Returns:
            Time token (e.g. self._get_time_token(1200, 1000) -> "<|0.20|>")
        """
        if time < segment_start or segment_start + DURATION < time:
            raise ValueError(
                f"Time {format_timestamp(time / 1000)} is out of the segment "
                f"({format_timestamp(segment_start / 1000)} - "
                f"{format_timestamp((segment_start + DURATION) / 1000)}) of {audio_path}"
            )

        time_in_segment = time - segment_start
        nearest_timestamp = (
            round(time_in_segment / self.timestamp_resolution)
            * self.timestamp_resolution
        )  # in milliseconds
        time_token = f"<|{nearest_timestamp / 1000:.2f}|>"
        return time_token

    def _get_prompt(self, prompt_buffer: Deque[PromptNode]) -> str:
        prompt_length = 0
        prompt_buffer_idx = len(prompt_buffer)
        while prompt_buffer_idx >= 1 and prompt_length < self.max_prompt_length:
            prompt_buffer_idx -= 1
            prompt_length += prompt_buffer[prompt_buffer_idx].num_tokens

        for _ in range(prompt_buffer_idx):
            prompt_buffer.popleft()

        return "".join([node.text for node in prompt_buffer])

    @staticmethod
    def read_records(path: Union[str, Path]) -> List[Record]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                record = Record(
                    audio_path=data["audio_path"],
                    text=data["text"],
                    language=data["language"],
                    prompt=data["prompt"],
                )
                records.append(record)
        return records

    @staticmethod
    def write_records(records: List[Record], path: Union[str, Path]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                data = {
                    "audio_path": record.audio_path,
                    "text": record.text,
                    "language": record.language,
                    "prompt": record.prompt,
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _subsample_silence(self) -> None:
        records = self.read_records(self.output)

        silence_records = filter(lambda record: record.text == "", records)
        non_silence_records = filter(lambda record: record.text != "", records)
        filtered_records = (
            list(non_silence_records)
            + list(silence_records)[:: self.subsampling_factor_for_silence]
        )

        Path(self.output).unlink()
        self.write_records(filtered_records, self.output)
