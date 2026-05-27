from transformers import AutoTokenizer
from settings.settings import model_settings

class TokenizerService:
    def __init__(self):
        self.tokenizer = None

    def load(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_settings.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

    def encode(self, text: str, return_tensors: bool = False):
        """Encode text into tokens.

        Args:
            text: Input string to tokenize.
            return_tensors: If True, return the raw tokenizer output with tensors
                (same shape as `tokenizer(...)`). If False (default), return a
                simple Python list of token ids for easy downstream usage and
                for unit tests that expect a list/tuple.
        """
        if self.tokenizer is None:
            self.load()
        assert self.tokenizer is not None
        if return_tensors:
            return self.tokenizer(text, return_tensors="pt", truncation=True, max_length=model_settings.max_length)

        # Default behavior: return plain Python list of token ids
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=model_settings.max_length)
        input_ids = encoded.get("input_ids")
        if input_ids is None:
            return []
        try:
            return input_ids[0].tolist()
        except Exception:
            return input_ids
    
    def decode(self, tokens):
        if self.tokenizer is None:
            self.load()
        assert self.tokenizer is not None
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
tokenizer_service = TokenizerService()