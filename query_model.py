"""
This is boilerplate for loading + querying LORA models.
(next step after this is gguf conversion)
"""

from unsloth import FastLanguageModel
import argparse

model_name = "lora_model"

max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# alpaca_prompt = You MUST copy from above!
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def query_model(query: str):
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            query, # instruction
            "", # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    tokenized_output = tokenizer.batch_decode(outputs)
    parsed_output = tokenized_output[0].split("### Response:")[1].strip()
    print("INPUT===============")
    print(query)
    print("INPUT===============")
    print("OUTPUT===============")
    print(parsed_output)
    print("OUTPUT===============")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="The query to send to the model")
    args = parser.parse_args()
    query = args.query
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", 
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    query_model(query)


