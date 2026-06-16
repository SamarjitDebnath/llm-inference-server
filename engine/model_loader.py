import torch
from transformers import AutoModelForCausalLM
from settings.settings import model_settings
from tokenizer.tokenizer_service import tokenizer_service
from logger import setup_logger
from settings.settings import logging_settings

logger = setup_logger(__name__, level=logging_settings.log_level, log_file=logging_settings.log_file)


class ModelLoader:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            device = model_settings.device
            logger.info("Loading model '%s' onto device '%s'", model_settings.model_name, device)

            # MPS (Apple Silicon) works best with float16.
            # float32 models can be loaded and cast, but large models may OOM in float32 on MPS.
            dtype = torch.float16 if device == "mps" else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                model_settings.model_name,
                dtype=dtype,
            )
            self.model.to(device)
            self.model.eval()
            logger.info(
                "Model loaded successfully on device='%s' dtype='%s' (model_type: %s, vocab_size: %d)",
                device,
                dtype,
                type(self.model).__name__,
                self.model.config.vocab_size if hasattr(self.model, 'config') else 'unknown'
            )

    def _get_model(self):
        if self.model is None:
            self.load()
        return self.model
    
    def warmup(self):
        model = self._get_model()
        
        if model is None:
            raise RuntimeError("Model failed to load during warmup")

        # Request tensor outputs for model warmup
        tokens = tokenizer_service.encode("Warmup request", return_tensors=True)
        input_ids = tokens["input_ids"].to(model_settings.device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            torch.argmax(logits, dim=-1)

model_loader = ModelLoader()