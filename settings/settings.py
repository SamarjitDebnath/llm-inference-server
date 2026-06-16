import torch
from utils.utils import *
from pydantic_settings import BaseSettings


def resolve_device(configured: str) -> str:
    """Return the best available compute device.

    If ``configured`` is ``"auto"``, auto-detect in priority order:
    1. MPS  - Apple Silicon / Metal (torch.backends.mps.is_available)
    2. CUDA - NVIDIA GPU           (torch.cuda.is_available)
    3. CPU  - universal fallback

    Any other value is returned as-is so the operator can always pin a
    specific device (e.g. ``"cpu"``, ``"cuda:1"``, ``"mps"``).
    """
    if configured != "auto":
        return configured
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ModelSetting:
    def __init__(self, config_path: str = "settings/config.yaml"):
        config = Utils.load_config(config_path)["model_config"]["defaults"]

        self.model_name = config["model_name"]
        self.device = resolve_device(config["device"])
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
