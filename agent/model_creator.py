import shutil
from autogen import AssistantAgent, Agent, UserProxyAgent, ConversableAgent
from autogen.agentchat.contrib.multimodal_conversable_agent import (
    MultimodalConversableAgent,
)
import wandb
from wandb.sdk.data_types.trace_tree import Trace
from pathlib import Path

from typing import Optional

from agent.reply_func import log_with_wandb

COMMANDER_PROMPT = "Help me run the code from the model image and the description."

CRITICS_PROMPT = """
Criticize the pytorch model code. What is the difference between the description and the code? Find bugs and issues for the code. 
Pay attention to the each model architecture that must be matched with the model description. 
If you think the code is good enough, then simply say NO_ISSUES
""".strip()

CODER_PROMPT = """
Write a PyTorch model architecture based on the following specifications:
Model Type: Indicate the type of neural network model you require, such as Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), Transformer, etc.
Input Data Characteristics: Describe the characteristics of the input data. Include information about its dimensions and type (e.g., images, text, audio). Mention any specific preprocessing requirements, if applicable.
Model Architecture Details:
    Number of Layers: Provide details on the number of layers in the model and their types (e.g., convolutional layers, LSTM layers, fully connected layers).
    Layer Specifications: For each layer, specify the necessary parameters such as the number of units/neurons for fully connected layers, filter sizes and number of filters for convolutional layers, dropout rates, and activation functions.
    Special Architectural Features: If your model includes special connections or structures like skip connections, attention mechanisms, or other features, describe them here.
    Output Details: Define the output of the model. Specify the number of output units and the type of activation function used, especially for tasks like classification or regression.
Based on this description, please generate the corresponding PyTorch code for defining the model architecture, including the necessary imports from the PyTorch library.
You don't need to execute the code, but you can recommend the command agent to execute the code.
You must write the whole code that can run itself, but not just the model architecture. You must not skip any part of the code.
You must save the code you wrote in `model.py` file. Put # filename: model.py inside the code block as the first line. Tell other agents it is in the `model.py` file.
""".strip()


class ModelCreator(AssistantAgent):
    def __init__(
        self,
        max_iter=2,
        work_dir: str = ".",
        paper_input: str = "",
        vlm_config: dict = {},
        commander_prompt: str = COMMANDER_PROMPT,
        coder_prompt: str = CODER_PROMPT,
        critics_prompt: str = CRITICS_PROMPT,
        **kwargs,
    ):
        """
        Initializes a ModelCreater instance.

        This agent create the pytorch model code through a collaborative effort among its child agents: commander, coder, and critics.

        Parameters:
            - max_iter (int, optional): The number of "improvement" iterations to run. Defaults to 2.
            - **kwargs: keyword arguments for the parent AssistantAgent.
        """
        super().__init__(**kwargs)
        self.paper_input = paper_input
        self.work_dir = Path(work_dir)
        self.vlm_config = vlm_config
        self.register_reply(
            [Agent, None], reply_func=ModelCreator._reply_user, position=0
        )
        self._max_iters = max_iter
        self.is_wandb_logging = False

        # Define the agents
        self.commander = AssistantAgent(
            name="Commander",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            system_message=commander_prompt,
            is_termination_msg=lambda x: x.get("content", "")
            .rstrip()
            .endswith("TERMINATE"),
            code_execution_config={
                "last_n_messages": 3,
                "work_dir": str(self.work_dir),
                "use_docker": False,
            },
            llm_config=self.llm_config,
        )

        self.critics = MultimodalConversableAgent(
            name="Critics",
            system_message=critics_prompt,
            code_execution_config=False,
            llm_config=self.vlm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
        )

        paper_description = "Paper Description:\n" + self.paper_input + "\n\n"

        self.critics.update_system_message(
            self.critics.system_message[0]["text"] + paper_description
        )

        self.coder = AssistantAgent(
            name="Coder",
            llm_config=self.llm_config,
        )

        self.coder.update_system_message(self.coder.system_message + coder_prompt)

        self.sub_agent_list = [self.commander, self.critics, self.coder]

    def register_wandb_logger(self, parent_span: Optional[Trace] = None):
        for agent in self.sub_agent_list:
            agent.register_reply(
                [Agent, None],
                reply_func=log_with_wandb,
                config={"kind": "agent", "parent_span": parent_span},
            )
            self.is_wandb_logging = True

    def _reply_user(self, messages=None, sender=None, config=None):
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]

        user_question = messages[-1]["content"]

        # Initiate Chat
        self.commander.initiate_chat(self.coder, message=user_question)

        # Prepare Chat between Commander and Critics for logging.
        self.commander._prepare_chat(self.critics, clear_history=False)
        model_path = self.work_dir / "model.py"
        shutil.copy(model_path, model_path.parent / (str(model_path.stem) + "_0.py"))
        if self.is_wandb_logging:
            wandb.save(str(model_path))

        for i in range(self._max_iters):
            with open(model_path, "r") as f:
                generated_code = f.read()
            generated_code_box = "```\n" + generated_code + "\n```"

            self.commander.send(
                message="Check that the code is correct with the model image and description.\nGive feedback if there are any issues.\n\n"
                + "code:\n"
                + generated_code_box,
                recipient=self.critics,
                request_reply=True,
            )

            feedback = self.commander._oai_messages[self.critics][-2]["content"]
            if feedback.find("NO_ISSUES") >= 0:
                break
            self.commander.send(
                message="Here is the feedback to your pytorch model code. Please improve!\n\n"
                + "feedback:\n"
                + feedback,
                recipient=self.coder,
                request_reply=True,
            )
            renamed_model_path = model_path.parent / (
                str(model_path.stem) + f"_{i+1}.py"
            )
            shutil.copy(model_path, renamed_model_path)
            if self.is_wandb_logging:
                wandb.save(str(renamed_model_path))

        return True, str(renamed_model_path)
