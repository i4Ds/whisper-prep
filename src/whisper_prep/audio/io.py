from pathlib import Path
from typing import Optional, Union

from pydub import AudioSegment, effects


def read_audio(
    path: Union[str, Path], resample_rate: Optional[int] = None, normalize: bool = True
) -> AudioSegment:
    format = Path(path).suffix[1:]
    audio_segment = AudioSegment.from_file(path, format=format)

    if resample_rate is not None:
        audio_segment = audio_segment.set_frame_rate(resample_rate)

    if normalize:
        audio_segment = effects.normalize(audio_segment)

    return audio_segment


def save_audio_segment(
    audio_segment: AudioSegment, path: Union[str, Path], format: str
) -> None:
    audio_segment.export(path, format=format)
