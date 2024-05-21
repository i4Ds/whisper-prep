import collections
from dataclasses import dataclass
from typing import Iterator, List, Tuple, Union

import numpy as np
import torch
import torchaudio
from webrtcvad import Vad


@dataclass
class Frame:
    bytes: bytes
    timestamp: float
    duration: float


class VoiceActivityDetector:
    def __init__(
        self,
        aggressiveness: int,
        frame_duration_ms: int,
        padding_duration_ms: int,
        rate_voiced_frames_threshold: float = 0.9,
        rate_unvoiced_frames_threshold: float = 0.9,
    ) -> None:
        self.vad = Vad(aggressiveness)

        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.num_padding_frames = int(self.padding_duration_ms / self.frame_duration_ms)

        self.rate_voiced_frames_threshold = rate_voiced_frames_threshold
        self.rate_unvoiced_frames_threshold = rate_unvoiced_frames_threshold

    def frame_generator(self, audio: bytes, sample_rate: int) -> Iterator[Frame]:
        n = int(sample_rate * (self.frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset : offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def return_start_end_of_audio(
        self,
        audio: np.array,
        sample_rate: int,
    ) -> tuple[float, float]:
        ring_buffer = collections.deque(maxlen=self.num_padding_frames)
        triggered = False

        start_second = None
        end_second = None

        # Get raw data
        raw_data = audio.raw_data

        for frame in self.frame_generator(raw_data, sample_rate):
            is_speech = self.vad.is_speech(frame.bytes, sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])

                if num_voiced > self.rate_voiced_frames_threshold * ring_buffer.maxlen:
                    triggered = True

                    if start_second is None:
                        start_second = ring_buffer[0][0].timestamp

                    ring_buffer.clear()
            else:
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])

                if (
                    num_unvoiced
                    > self.rate_unvoiced_frames_threshold * ring_buffer.maxlen
                ):
                    triggered = False
                    ring_buffer.clear()
                    end_second = frame.timestamp

        if triggered:
            end_second = (len(raw_data) / sample_rate) / 2.0

        if start_second is None:
            start_second = 0

        return start_second, end_second


class VADSilero:
    def __init__(
        self,
        model_repo: str = "snakers4/silero-vad",
        model_name: str = "silero_vad",
        force_reload: bool = True,
    ):
        self.model, self.utils = torch.hub.load(
            repo_or_dir=model_repo, model=model_name, force_reload=force_reload
        )
        self.get_speech_timestamps = self.utils[0]

    def return_start_end_of_audio(
        self, audio: Union[str, np.array], sample_rate: int = 16000
    ) -> Tuple[float, float]:
        list_start_ends = self.return_speech_timestamps(audio, sample_rate)
        return list_start_ends[0]["start"], list_start_ends[-1]["end"]

    def return_speech_timestamps(
        self, audio: Union[str, np.array], sample_rate: int = 16000
    ) -> List[Tuple[float, float]]:
        if isinstance(audio, str):
            wav, aud_sr = torchaudio.load(audio)
            wav = torchaudio.functional.resample(
                wav, orig_freq=aud_sr, new_freq=sample_rate
            )
            wav = torch.mean(wav, dim=0)
        else:
            wav, sample_rate = torch.from_numpy(audio), sample_rate

        # get speech timestamps from full audio file
        speech_timestamps = self.get_speech_timestamps(
            wav, self.model, sampling_rate=sample_rate, return_seconds=True
        )
        return [(ts["start"], ts["end"]) for ts in speech_timestamps]
