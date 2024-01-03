import io
import os
from pathlib import Path
import re
import sys
import datetime

import wandb
import yaml

DEFAULT_CONFIG_PATH = "./config/wandb_config.yaml"


class MultiStream(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            if isinstance(stream, io.FileIO):
                stream.write(data.encode("utf-8"))
            else:
                stream.write(data)

    def flush(self):
        for stream in self.streams:
            if hasattr(stream, "flush"):
                stream.flush()


def convert_img_src_to_absolute(text, data_dir: str) -> str:
    # Extract the relative path from the img tag
    relative_path = re.search(r"<img (.*?)>", text).group(1)

    # Join the working directory path with the relative path to get the absolute path
    absolute_path = os.path.abspath(os.path.join(data_dir, relative_path))

    # Replace the relative path in the img tag with the absolute path
    converted_text = text.replace(relative_path, absolute_path)

    return converted_text


def preprocess_script(script: str, use_image: bool = True, data_dir: str = "./") -> str:
    if use_image:
        script = convert_img_src_to_absolute(script, data_dir)
    else:
        # Define the regex pattern for the <img> tag
        pattern = r"<img .+?>"
        script = re.sub(pattern, "", script)
    return script


def attach_file2stdout(file_dir: Path) -> MultiStream:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = file_dir / f"chat_log_{current_time}.log"
    file_stream = io.FileIO(log_file_path, "w")

    multi_stream = MultiStream(sys.stdout, file_stream)
    sys.stdout = multi_stream
    return multi_stream


def load_config(config_path: str, run_path: str):
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif run_path:
        api = wandb.Api()
        run = api.run(run_path)
        config = run.config
    else:
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
    return config
