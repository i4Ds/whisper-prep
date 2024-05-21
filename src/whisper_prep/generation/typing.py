from dataclasses import dataclass
from typing import Optional


@dataclass
class Utterance:
    """
    Representing a single segment of audio with a transcription. Corresponds to a single chunk in a
    .srt (or .vtt) file.
    """

    text: str
    start: Optional[int] = None  # in milliseconds
    end: Optional[int] = None  # in milliseconds


@dataclass
class Record:
    """
    A single training instance for Whisper.
    `text` can include timestamps in the format of <|0.00|>.
    """

    audio_path: str
    text: str  # text including timestamps
    language: str = "de"
    prompt: str = ""  # previous text including timestamps


@dataclass
class PromptNode:
    text: str  # text including timestamps
    num_tokens: int
