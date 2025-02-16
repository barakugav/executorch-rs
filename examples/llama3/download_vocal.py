import json

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
print(json.dumps(tokenizer.vocab))
