import io
import os
from pathlib import Path
import re
import sys
import datetime


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


def convert_img_src_to_absolute(text, workdir: str) -> str:
    # Extract the relative path from the img tag
    relative_path = re.search(r"<img (.*?)>", text).group(1)

    # Join the working directory path with the relative path to get the absolute path
    absolute_path = os.path.abspath(os.path.join(workdir, relative_path))

    # Replace the relative path in the img tag with the absolute path
    converted_text = text.replace(relative_path, absolute_path)

    return converted_text


def attach_file2stdout(file_dir: Path) -> MultiStream:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = file_dir / f"chat_log_{current_time}.log"
    file_stream = io.FileIO(log_file_path, "w")

    multi_stream = MultiStream(sys.stdout, file_stream)
    sys.stdout = multi_stream
    return multi_stream
