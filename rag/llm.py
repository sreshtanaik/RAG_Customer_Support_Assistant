"""The language model that writes the answer. Runs locally, no API key needed."""

from rag import config


class Flant5Generator:
    def __init__(self, model_name: str = config.LLM_MODEL):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def __call__(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output_ids = self.model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
