from datetime import datetime
import os
import autogen
import argparse
from pathlib import Path

import wandb
from agent import ModelCreator
from agent.reply_func import get_current_time, log_with_wandb
from utils.utils import attach_file2stdout, convert_img_src_to_absolute
from autogen.agentchat.contrib.multimodal_conversable_agent import (
    MultimodalConversableAgent,
)
from autogen import AssistantAgent, Agent, UserProxyAgent, ConversableAgent
from config import gpt4v_config, gpt4_config
from wandb.sdk.data_types.trace_tree import Trace

INTERPRETER_PROMPT = """
You are machine learning paper interpreter. 
You must interpret the paper and give the description of the model for the coder agent to make the model code better.
You have to carefully read the given text description about the model and extract information from the input images, if you need.
""".strip()

wandb.init(project="paper2code")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="The data directory path")
    parser.add_argument("--output_dir", type=str, help="The output directory path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    output_dir = Path(args.output_dir) / data_dir.name / current_time
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = data_dir / "script.txt"

    with open(script_path) as f:
        script = f.read()

    script = convert_img_src_to_absolute(script, data_dir)

    root_span = Trace(name="root", kind="agent")

    creator = ModelCreator(
        name="Model Creator",
        work_dir=str(output_dir),
        paper_input=script,
        vlm_config=gpt4v_config,
        llm_config=gpt4_config,
    )

    user_proxy = UserProxyAgent(
        name="User", human_input_mode="NEVER", max_consecutive_auto_reply=0
    )

    user_proxy.register_reply(
        [Agent, None],
        reply_func=log_with_wandb,
        config={"kind": "agent", "parent_span": root_span},
    )

    logging_stream = attach_file2stdout(output_dir)

    root_span._span.start_time_ms = get_current_time()

    interpreter = MultimodalConversableAgent(
        name="Interpreter",
        human_input_mode="NEVER",
        system_message=INTERPRETER_PROMPT,
        llm_config=gpt4v_config,
    )

    interpreter.register_reply(
        [Agent, None],
        reply_func=log_with_wandb,
        config={"kind": "agent", "parent_span": root_span},
    )

    user_proxy.send(message=script, recipient=interpreter, request_reply=True)

    interpret = user_proxy._oai_messages[interpreter][-1]["content"]

    # Create the creator span
    creator_span = Trace(name="creator", kind="chain")
    root_span.add_child(creator_span)

    # Register wandb log reply function
    creator.register_wandb_logger(parent_span=creator_span)

    creator_span._span.start_time_ms = get_current_time()

    user_proxy.initiate_chat(
        creator,
        message="Make a PyTorch model architecture based on the model description\n\ndescription:\n"
        + interpret,
    )
    creator_span._span.end_time_ms = get_current_time()
    root_span._span.end_time_ms = get_current_time()

    with open(output_dir / "model.py") as f:
        model_code = f.read()

    creator_span.add_inputs_and_outputs(
        inputs={"input": interpret if use_interpreter else script},
        outputs={"output": model_code},
    )
    root_span.add_inputs_and_outputs(
        inputs={"script": script},
        outputs={"result": user_proxy.last_message(agent=creator)},
    )

    root_span.log(name="paper2code")

    wandb.finish()
