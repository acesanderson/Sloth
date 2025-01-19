import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(api_token)

login(token=api_token)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", api_key=api_token
)
tokenizer.save_pretrained("./model/")

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model.save_pretrained("./model/")
