import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=config_path)
    return parser.parse_args()


def config_path(path: str):
    if Path(path).exists():
        return path
    else:
        raise argparse.ArgumentTypeError(f"Config path:{path} is not a valid path.")
