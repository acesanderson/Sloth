from Kramer.database.MongoDB_CRUD import get_all_courses_sync  # Replace with your actual import
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
import re
import html
from unsloth import FastLanguageModel
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments
from trl import SFTTrainer


# Data Preparation
def zap_instructors(description: str, instructor_name: str):
    try:
        return description.replace(instructor_name, "INSTRUCTOR_NAME")
    except:
        return description


def clean_html_text(text):
    try:
        text = html.unescape(text)  # Unescape HTML entities first
    except:
        pass # Handle cases where unescaping fails
    try:
        clean_text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
        return clean_text
    except:
        return text

courses = get_all_courses_sync()

course_titles = [clean_html_text(course.metadata['Course Name']) for course in courses]
course_descriptions = [clean_html_text(zap_instructors(course.metadata['Course Description'], course.metadata['Instructor Name'])) for course in courses]

dataset = Dataset.from_dict({"title": course_titles, "description": course_descriptions})

# Limit to 1,000 for testing purposes
# dataset = dataset.select(range(1000))

# Apply splits
splits = dataset.train_test_split(test_size=0.2, seed=42)
test_valid = splits["test"].train_test_split(test_size=0.5, seed=42)

dataset_dict = DatasetDict({
    "train": splits["train"],
    "validation": test_valid["train"],
    "test": test_valid["test"],
})


# Model initialization remains the same as in the original script
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = 512,  # Reduced since course descriptions are typically shorter
    dtype = None,
    load_in_4bit = True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Modified prompt template for course descriptions
course_prompt = """Generate a detailed course description for the following course title.

### Course Title:
{}

### Description:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    titles = examples["title"]
    descriptions = examples["description"]
    texts = []
    for title, description in zip(titles, descriptions):
        # Add EOS token to end of description
        text = course_prompt.format(title, str(description) + EOS_TOKEN)
        texts.append(text)
    return {"text": texts}

# Format all splits
train_formatted = dataset_dict["train"].map(formatting_prompts_func, batched=True)
val_formatted = dataset_dict["validation"].map(formatting_prompts_func, batched=True)

# Training configuration
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_formatted,
    eval_dataset = val_formatted,  # Added validation dataset
    dataset_text_field = "text",
    max_seq_length = 512,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        per_device_eval_batch_size = 4,  # Added eval batch size
        gradient_accumulation_steps = 4,
        warmup_steps = 100,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        evaluation_strategy = "steps",  # Added evaluation strategy
        eval_steps = 50,  # Evaluate every 50 steps
        save_strategy = "steps",
        save_steps = 50,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "course_description_model",
    ),
)

# Train the model
trainer_stats = trainer.train()

# Save the model
model.save_pretrained("course_description_lora")
tokenizer.save_pretrained("course_description_lora")

# Inference helper function
def generate_description(title, model, tokenizer):
    prompt = course_prompt.format(title, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        use_cache=True
    )
    return tokenizer.batch_decode(outputs)[0]

# Test the model on some examples from the test set
FastLanguageModel.for_inference(model)
test_examples = dataset_dict["test"].select(range(5))  # Test on 5 examples

for title in test_examples["title"]:
    print("\nOriginal Title:", title)
    generated_description = generate_description(title, model, tokenizer)
    print("\nGenerated Description:", generated_description)
    print("\n" + "="*80)
