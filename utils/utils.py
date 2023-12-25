import os
import re


def convert_img_src_to_absolute(text, workdir: str) -> str:
    # Extract the relative path from the img tag
    relative_path = re.search(r"<img (.*?)>", text).group(1)

    # Join the working directory path with the relative path to get the absolute path
    absolute_path = os.path.abspath(os.path.join(workdir, relative_path))

    # Replace the relative path in the img tag with the absolute path
    converted_text = text.replace(relative_path, absolute_path)

    return converted_text
