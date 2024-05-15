from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from string import Template
from typing import Union


@dataclass
class Caption:
    start_second: float
    end_second: float
    text: str
    offset: float = 0

    def start_timestamp(self) -> str:
        td = timedelta(seconds=self.start_second + self.offset)
        return format_delta(delta=td, pattern="{h}:{m}:{s}.{ms}")

    def end_timestamp(self) -> str:
        td = timedelta(seconds=self.end_second + self.offset)
        return format_delta(delta=td, pattern="{h}:{m}:{s}.{ms}")


def format_delta(delta: timedelta, pattern: str) -> str:
    d = {}
    d["h"], rem = divmod(delta.seconds, 3600)
    d["m"], d["s"] = divmod(rem, 60)
    d["ms"] = delta.microseconds // 1000

    d["h"] = f'{d["h"]:02d}'
    d["m"] = f'{d["m"]:02d}'
    d["s"] = f'{d["s"]:02d}'
    d["ms"] = f'{d["ms"]:03d}'

    return pattern.format(**d)


def generate_srt(captions: list[Caption], save_path: Union[str, Path]) -> None:
    srt_content = ""

    temp_str = "${index}\n${start} --> ${end}\n${text}\n\n"
    temp_obj = Template(temp_str)

    for idx, caption in enumerate(captions):
        srt_content += temp_obj.substitute(
            index=idx + 1,
            start=caption.start_timestamp(),
            end=caption.end_timestamp(),
            text=caption.text,
        )

    with open(save_path, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)
