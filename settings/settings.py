from utils.utils import *
from pydantic_settings import BaseSettings


class ModelSetting:
    def __init__(self, config_path: str = "settings/config.yaml"):
        config = Utils.load_config(config_path)["model_config"]["defaults"]

        self.model_name = config["model_name"]
        self.device = config["device"]
        self.max_length = config["max_length"]
        self.temperature = config["temperature"]
        self.top_k = config["top_k"]
        self.top_p = config["top_p"]
        self.repetition_penalty = config["repetition_penalty"]
        self.num_return_sequences = config["num_return_sequences"]

class LoggingSetting:
    def __init__(self, config_path: str = "settings/config.yaml"):
        config = Utils.load_config(config_path)["logging_config"]["defaults"]

        self.log_level = config["log_level"]
        self.log_file = config["log_file"]


class SecretSetting(BaseSettings):
    hf_key: str | None = ""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }

model_settings = ModelSetting()
logging_settings = LoggingSetting()
secret_settings = SecretSetting()
