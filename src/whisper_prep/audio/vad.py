import collections
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from webrtcvad import Vad

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

SILERO_MODEL = load_silero_vad()

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



def silero_vad_collector(
    path: str,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    window_size_samples: int = 1024,
    speech_pad_ms: int = 30,
) -> tuple[float, float]:

    audio = read_audio(path)

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        audio,
        SILERO_MODEL,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        window_size_samples=window_size_samples,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )

    if not speech_timestamps:
        return 0.0, 0.0

    start_second = speech_timestamps[0]["start"]
    end_second = speech_timestamps[-1]["end"]

    return start_second, end_second
