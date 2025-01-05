import argparse
from unsloth import FastLanguageModel
from pathlib import Path

dir_path = Path(__file__).parent
models_dir = dir_path.parent / "models"
model_name = "titles_to_tocs"

course_prompt = """
Look at this title of a LinkedIn Learning video course, and create a descriptive Table of Contents for the course.

### Description
{}


### Table of Contents:
{}"""


# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = str(models_dir / model_name),
    max_seq_length = 256,
    dtype = None,
    load_in_4bit = True,
)

# Enable faster inference mode
FastLanguageModel.for_inference(model)

# Inference helper function
def generate_TOC_from_title(course_title: str, model, tokenizer):
    prompt = course_prompt.format(course_title, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=.5, # Lowered from 0.7 for this one
        top_p=0.9,
        use_cache=True
    )
    return tokenizer.batch_decode(outputs)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Table of Contents for a LinkedIn Learning video course.")
    parser.add_argument("course_title", type=str, help="The title and description of the course.")
    args = parser.parse_args()
    generated_TOC = generate_TOC_from_title(args.course_title_and_description, model, tokenizer)
    print(generated_TOC)
    
