import os
import autogen
import argparse
from pathlib import Path
from agent import ModelCreator
from utils.utils import attach_file2stdout, convert_img_src_to_absolute
from autogen.agentchat.contrib.multimodal_conversable_agent import (
    MultimodalConversableAgent,
)
from config import gpt4v_config, gpt4_config

INTERPRETER_PROMPT = """
You are machine learning paper interpreter. 
You must interpret the paper and give the description of the model for the coder agent to make the model code better.
You have to carefully read the given text description about the model and extract information from the input images, if you need.
""".strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="The data directory path")
    parser.add_argument("--output_dir", type=str, help="The output directory path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / data_dir.name
    script_path = data_dir / "script.txt"

    with open(script_path) as f:
        script = f.read()

    script = convert_img_src_to_absolute(script, data_dir)

    creator = ModelCreator(
        name="Model Creator~",
        work_dir=str(output_dir),
        paper_input=script,
        vlm_config=gpt4v_config,
        llm_config=gpt4_config,
    )

    interpreter = MultimodalConversableAgent(
        name="Interpreter",
        human_input_mode="NEVER",
        system_message=INTERPRETER_PROMPT,
        llm_config=gpt4v_config,
    )

    user_proxy = autogen.UserProxyAgent(
        name="User", human_input_mode="NEVER", max_consecutive_auto_reply=0
    )
    logging_stream = attach_file2stdout(output_dir)

    user_proxy.send(message=script, recipient=interpreter, request_reply=True)

    interpret = user_proxy._oai_messages[interpreter][-1]["content"]

    user_proxy.initiate_chat(
        creator,
        message="Make a PyTorch model architecture based on the model description\n\ndescription:\n"
        + interpret,
    )
