import re
from wandb.sdk.data_types.trace_tree import Trace
import datetime

SPAN_NAME = "paper2code"


def get_current_time():
    return round(datetime.datetime.now().timestamp() * 1000)


def check_exit_code(message: str):
    # Search for the exit code
    if isinstance(message, str):
        match = re.search(r"exitcode: (\d+)", message)

        if match:
            # If a match was found, print the exit code
            exitcode = int(match.group(1))
            return exitcode
        else:
            return None
    else:
        return None


def log_with_wandb(recipient, messages, sender, config):
    parent_span = config["parent_span"] if config["parent_span"] is not None else None
    cur_time = get_current_time()
    recipient.span_dict = getattr(recipient, "span_dict", {})
    recipient.span_dict[sender.name] = Trace(
        name=recipient.name,
        kind=config["kind"],
        start_time_ms=cur_time,
        metadata={
            "system_prompt": recipient.system_message,
            "llm_config": recipient.llm_config,
        },
    )
    if hasattr(sender, "span_dict") and recipient.name in sender.span_dict:
        sender_span = sender.span_dict[recipient.name]
        sender_span._span.end_time_ms = cur_time
        receive_message = messages[-2]["content"]
        reply_message = messages[-1]["content"]
        if isinstance(receive_message, str):
            receive_message = "\n" + receive_message
        if isinstance(reply_message, str):
            reply_message = "\n" + reply_message
        sender.span_dict[recipient.name].add_inputs_and_outputs(
            inputs={"receive_message": receive_message},
            outputs={"reply_message": reply_message},
        )
        code_exit_code = check_exit_code(reply_message)
        if code_exit_code is not None:
            if code_exit_code == 0:
                sender_span._span.status_code = "SUCCESS"
            else:
                sender_span._span.status_code = "ERROR"
            sender_span.log(name=SPAN_NAME)

    if parent_span:
        parent_span.add_child(recipient.span_dict[sender.name])
    return False, None
