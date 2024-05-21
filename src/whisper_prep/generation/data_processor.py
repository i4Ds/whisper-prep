import json
import unicodedata
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Union

import torch
import torchaudio
from tqdm import tqdm
from whisper.audio import load_audio
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from whisper.utils import format_timestamp

from whisper_prep.generation.typing import PromptNode, Record, Utterance

DURATION = 30000  # 30 seconds in milliseconds
SAMPLE_RATE = 16000
DURATION_IN_SAMPLES = int(DURATION * SAMPLE_RATE / 1000)


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
        max_prompt_length: int = 223
        * 1.2,  # 223 tokens and some extra for the time stamps.
        max_tokens_length: int = 219,
        subsampling_factor_for_silence: int = 1,
        rep_threshold: int = 3,
        tokenizer_type: str = "multilingual",
        normalize_unicode: bool = False,
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
        self.rep_threshold = rep_threshold
        self.tokenizer_type = tokenizer_type
        self.normalize_unicode = normalize_unicode

        self._verify_args()

        self.tokenizer = get_tokenizer(
            multilingual=(self.tokenizer_type == "multilingual")
        )
        Path(self.dump_dir).mkdir(parents=True, exist_ok=True)

    def _verify_args(self) -> None:
        if self.with_timestamps:
            if not self.audio_dir or not self.transcript_dir:
                raise ValueError(
                    "`audio_dir` and `transcript_dir` must be set when `with_timestamps` is True"
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

    def run(self) -> None:
        if self.with_timestamps:
            self._process_with_timestamps()
        else:
            self._process_without_timestamps()

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
        if not utterances:
            return []

        # This function assumes utterances are already sorted by start time.
        result_utterances = []
        repeat_count = 1
        last_text = None

        for i in range(len(utterances)):
            current_text = utterances[i].text

            if i == 0:
                last_text = current_text
                continue

            if current_text == last_text:
                repeat_count += 1
            else:
                if repeat_count < threshold:
                    # Append all non-repeated or less than threshold utterances
                    result_utterances.extend(utterances[i - repeat_count : i])
                # Reset count and update last_text
                repeat_count = 1
                last_text = current_text

        # Check the last sequence at the end of the list
        if repeat_count < threshold:
            result_utterances.extend(utterances[-repeat_count:])

        return result_utterances

    def _drop_single_letter_utterance(
        self, utterances: List[Utterance]
    ) -> List[Utterance]:
        sanitized_utterances = []
        for utterance in utterances:
            if len(utterance.text.strip()) > 0:
                sanitized_utterances.append(utterance)

        return sanitized_utterances

    def _sanitize_utterances(self, utterances: List[Utterance]) -> List[Utterance]:
        if not utterances:
            return []

        # Remove duplicate hallucinations
        utterances = self._drop_repeated_utterances(utterances)

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
        audio_paths = list(Path(self.audio_dir).iterdir())

        for audio_path in tqdm(audio_paths):
            speech_id = audio_path.stem
            transcript_found = False

            for format in self.transcript_formats:
                transcript_path = Path(self.transcript_dir) / format.format(
                    id=speech_id
                )
                if transcript_path.exists():
                    try:
                        if transcript_path.suffix == ".srt":
                            utterances_for_speech = self.read_utterances_from_srt(
                                transcript_path, self.normalize_unicode
                            )
                        elif transcript_path.suffix == ".vtt":
                            utterances_for_speech = self.read_utterances_from_vtt(
                                transcript_path, self.normalize_unicode
                            )
                        # Sanitize utterances, if necessary.
                        # Takes care of some random timestamps error produces by the VAD of whisperx.
                        if not self._is_valid_utterances(utterances_for_speech, 0):
                            utterances_for_speech = self._sanitize_utterances(
                                utterances_for_speech
                            )
                        records = self._create_records_with_timestamps(
                            utterances_for_speech, audio_path
                        )
                        self.write_records(records, self.output)
                        transcript_found = True
                        break
                    except Exception as e:
                        print(e)
                        print(
                            f"Skipping {transcript_path} due to an error in the transcript"
                        )
                        continue

            if not transcript_found:
                raise FileNotFoundError(f"Transcript file not found for {speech_id}")

    @staticmethod
    def read_utterances_from_srt(
        transcript_path: Union[str, Path], normalize_unicode: bool = False
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

                start_time, end_time = lines[utterance_start].strip().split(" --> ")
                start_time = DataProcessor.str_to_milliseconds(start_time)
                end_time = DataProcessor.str_to_milliseconds(end_time)

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
                if normalize_unicode:
                    text = unicodedata.normalize("NFKC", text)
                if text == "":
                    # With time-aligned data, empty utterances will be created from timestamps later
                    # and are not necessary in the first place
                    continue

                utterances.append(Utterance(text=text, start=start_time, end=end_time))

        return utterances

    @staticmethod
    def read_utterances_from_vtt(
        transcript_path: Union[str, Path], normalize_unicode: bool = False
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

                start_time, end_time = lines[utterance_start].strip().split(" --> ")
                start_time = DataProcessor.str_to_milliseconds(start_time)
                end_time = DataProcessor.str_to_milliseconds(end_time)

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
                if normalize_unicode:
                    text = unicodedata.normalize("NFKC", text)
                if text == "":
                    # With time-aligned data, empty utterances will be created from timestamps later
                    # and are not necessary in the first place
                    continue

                utterances.append(Utterance(text=text, start=start_time, end=end_time))

        return utterances

    def _create_records_with_timestamps(
        self, utterances: List[Utterance], audio_path: Path
    ) -> List[Record]:
        audio = torch.tensor(load_audio(audio_path))
        dump_dir = Path(self.dump_dir) / audio_path.stem
        dump_dir.mkdir(parents=True, exist_ok=True)
        records = []
        prompt_buffer: Deque[PromptNode] = deque()
        segment_start, segment_end = 0, DURATION  # in milliseconds

        idx = 0
        while idx < len(utterances):
            # If the utterance is included in the segment and longer than the segment, skip it.
            if (
                utterances[idx].start < segment_end
                and utterances[idx].start + DURATION < utterances[idx].end
            ):
                segment_start = utterances[idx].end
                segment_end = segment_start + DURATION
                idx += 1
                continue

            segment_audio_path = self._save_segment_audio(
                audio, segment_start, dump_dir
            )
            prompt = self._get_prompt(prompt_buffer)

            segment_utterances = []
            while idx < len(utterances) and utterances[idx].start < segment_end:
                segment_utterances.append(utterances[idx])
                idx += 1

            if not self._is_valid_utterances(segment_utterances, segment_start):
                tqdm.write(
                    f"Skipping {audio_path} ({format_timestamp(segment_start / 1000)}-"
                    f"{format_timestamp(segment_end / 1000)}) because it contains invalid "
                    f"utterances: {segment_utterances}"
                )
                prompt_buffer.clear()
                segment_start = max(segment_end, segment_utterances[-1].end)
                segment_end = segment_start + DURATION
                continue

            tokens_length = 0
            segment_text = []
            for utterance in segment_utterances:
                start_token = self._get_time_token(
                    utterance.start, segment_start, audio_path
                )
                if utterance.end <= segment_end:
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
                else:
                    segment_text.append(start_token)
                    new_prompt_node = PromptNode(start_token, 1)
                    tokens_length += 1

                prompt_buffer.append(new_prompt_node)

            if tokens_length > self.max_tokens_length:
                tqdm.write(
                    f"Skipping {audio_path} ({format_timestamp(segment_start / 1000)}-"
                    f"{format_timestamp(segment_end / 1000)}) because it is too long "
                    f"({tokens_length} tokens)"
                )
            else:
                record = Record(
                    audio_path=segment_audio_path,
                    language=self.language,
                    text="".join(segment_text),
                    prompt=prompt,
                )
                records.append(record)

            if len(segment_utterances) == 0:
                segment_start += DURATION
            elif segment_utterances[-1].end <= segment_end:
                segment_start = segment_utterances[-1].end
            else:  # segment_utterances[-1].end > segment_end
                # The text of the last utterance was not included in the segment and will be
                # included in the next segment
                segment_start = segment_utterances[-1].start
                idx -= 1
            segment_end = segment_start + DURATION

        return records

    def _save_segment_audio(
        self, audio: torch.Tensor, segment_start: int, dump_dir: Path
    ) -> str:
        audio_start_idx = int(segment_start * SAMPLE_RATE / 1000)
        segment_audio_path = str((dump_dir / f"{segment_start}.mp3").absolute())
        segment_audio = audio[
            audio_start_idx : min(audio_start_idx + DURATION_IN_SAMPLES, audio.size(0))
        ]
        torchaudio.save(
            segment_audio_path, segment_audio.unsqueeze(0), SAMPLE_RATE, encoding="mp3"
        )
        return segment_audio_path

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
