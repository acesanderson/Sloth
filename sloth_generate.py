from unsloth import FastLanguageModel
import torch
import html
import re

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "course_description_lora",
    max_seq_length = 256,
    dtype = None,
    load_in_4bit = True,
)

# Enable faster inference mode
FastLanguageModel.for_inference(model)

# Define our course prompt template
course_prompt = """Generate a detailed course description for the following course title.

### Course Title:
{}

### Description:
{}"""

def clean_text(text: str) -> str:
    """Clean text by removing HTML entities, special tokens, and template text."""
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Fix common UTF-8 issues
    text = text.replace("â€™", "'")
    text = text.replace("â€\"", "-")
    text = text.replace("â€œ", '"')
    text = text.replace("â€", '"')
    
    # Remove special tokens
    text = text.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "")
    
    # Remove any remaining HTML-like tags
    text = re.sub(r'<[^>]+>', '', text)
    
    return text.strip()

def generate_description(title, model, tokenizer):
    prompt = course_prompt.format(title, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        use_cache=True
    )
    
    # Get raw output
    output_text = tokenizer.batch_decode(outputs)[0]
    
    # Remove the template
    template_start = "### Description:"
    description_start = output_text.find(template_start) + len(template_start)
    clean_output = output_text[description_start:].strip()
    
    # Remove the input prompt if it appears
    if "Generate a detailed course description" in clean_output:
        prompt_end = clean_output.find("### Description:")
        if prompt_end != -1:
            clean_output = clean_output[prompt_end + len("### Description:"):].strip()
    
    # Clean HTML and special characters
    return clean_text(clean_output)

if __name__ == "__main__":
    # Test titles
    test_titles = [
        "Hugging Face for Losers",
        "Car Repair for Recent Immigrants",
        "Navigating the Cape of Good Hope"
    ]
    
    print("Generating descriptions for test titles...")
    print("-" * 80)
    
    for title in test_titles:
        print(f"\nTitle: {title}")
        print("\nGenerated Description:")
        description = generate_description(title, model, tokenizer)
        print(description)
        print("-" * 80)
