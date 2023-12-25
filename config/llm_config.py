from pathlib import Path
import autogen

current_dir = Path(__file__).resolve().parent

oai_config_path = str(current_dir / "OAI_CONFIG_LIST.json")

config_list_4v = autogen.config_list_from_json(
    oai_config_path,
    filter_dict={
        "model": ["gpt-4-vision-preview"],
    },
)

config_list_gpt4 = autogen.config_list_from_json(
    oai_config_path,
    filter_dict={
        "model": ["gpt-4-1106-preview"],
    },
)

gpt4_config = {"config_list": config_list_gpt4, "max_tokens": 1000, "cache_seed": 42}
gpt4v_config = {"config_list": config_list_4v, "max_tokens": 1000, "cache_seed": 42}
