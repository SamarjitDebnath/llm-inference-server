import torch
from transformers import AutoModelForCausalLM
from settings.settings import model_settings
from tokenizer.tokenizer_service import tokenizer_service

class ModelLoader:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(model_settings.model_name)
            self.model.to(model_settings.device)
            self.model.eval()

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