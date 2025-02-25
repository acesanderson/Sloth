from datasets import Dataset, DatasetDict
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
import html
from unsloth import FastLanguageModel
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments
from trl import SFTTrainer
from pathlib import Path
import socket

# Our models directory
dir_path = Path(__file__).parent
models_dir = dir_path.parent / "models"

# Configs
## Get hostname
hostname = socket.gethostname()
configs = {}
if hostname == "Botvinnik":
    # configs['base_model'] = "unsloth/mistral-7b" 
    configs['base_model'] = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
    # configs['base_model'] = "unsloth/llama-2-7b"
    configs['model_max_seq_length'] = 128 
    configs['per_device_train_batch_size'] = 1
    configs['per_device_eval_batch_size'] = 1
    configs['gradient_accumulation_steps'] = 8
    configs['training_max_seq_length'] = 256
    configs['gradient_checkpointing'] = True
if hostname == "Caruana":
    configs['base_model'] = "unsloth/Meta-Llama-3.1-70B-bnb-4bit"
    # configs['base_model']="unsloth/Meta-Llama-3.1-8B"
    configs['model_max_seq_length'] = 512
    configs['per_device_train_batch_size'] = 4
    configs['per_device_eval_batch_size'] = 4
    configs['gradient_accumulation_steps'] = 4
    configs['training_max_seq_length'] = 1024
    configs['gradient_checkpointing'] = False


# Training function -- we'll want to set this apart from dataset logic, in the training directory
def train_sloth(model_name: str, prompt_template: str, data: dict):
    """
    Fine tune a model on a given dataset using the unsloth library.
    model_name: str - The name of the fine tuned model to save (this is saved in sloth/models)
    prompt_template: str - The prompt template to use for fine tuning
    data: dict - The data to use for fine tuning. The dict should have two keys: "inputs" and "outputs"
    """
    model_path = models_dir / model_name
    dataset = Dataset.from_dict(data)
    # Limit to 1,000 for testing purposes
    # dataset = dataset.select(range(80))
    # Apply splits
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = splits["test"].train_test_split(test_size=0.5, seed=42)

    dataset_dict = DatasetDict(
        {
            "train": splits["train"],
            "validation": test_valid["train"],
            "test": test_valid["test"],
        }
    )

    # Peek at the dataset to make sure everything is right -- print a couple samples
    print("## PEEK")
    print(dataset_dict["train"][0])
    print("## PEEK")

    # Model initialization remains the same as in the original script
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=configs['base_model'],
        max_seq_length=configs['model_max_seq_length'],  # Reduced since course descriptions are typically shorter
        dtype=None,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        inputs = examples["inputs"]
        outputs = examples["outputs"]
        texts = []
        for input, output in zip(inputs, outputs):
            # Add EOS token to end of description
            text = prompt_template.format(input, str(output) + EOS_TOKEN)
            texts.append(text)
        return {"text": texts}

    # Format all splits
    train_formatted = dataset_dict["train"].map(formatting_prompts_func, batched=True)
    val_formatted = dataset_dict["validation"].map(
        formatting_prompts_func, batched=True
    )

    # Training configuration
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_formatted,
        eval_dataset=val_formatted,  # Added validation dataset
        dataset_text_field="text",
        max_seq_length=configs['training_max_seq_length'],
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=configs['per_device_train_batch_size'],
            per_device_eval_batch_size=configs['per_device_eval_batch_size'],  # Added eval batch size
            gradient_accumulation_steps=configs['gradient_accumulation_steps'],
            warmup_steps=100,
            num_train_epochs=5,  # 3 for the courrse title task, 5 for this one
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            evaluation_strategy="steps",  # Added evaluation strategy
            eval_steps=50,  # Evaluate every 50 steps
            save_strategy="steps",
            save_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(model_path),
            gradient_checkpointing = configs['gradient_checkpointing'],
            save_total_limit=2,     # Keep only the last 2 checkpoints
            resume_from_checkpoint=True,
        ),
    )

    # Train the model
    trainer_stats = trainer.train()

    # Save the model
    model.save_pretrained(str(model_path))
    tokenizer.save_pretrained(str(model_path))

    # Test the model on some examples from the test set
    FastLanguageModel.for_inference(model)
    test_examples = dataset_dict["test"].select(range(5))  # Test on 5 examples

    for input in test_examples["inputs"]:
        print("\nOriginal Input:", input)
        output = query_model(input = input, prompt_template = prompt_template, model = model, tokenizer = tokenizer)
        print("\nGenerated Output:", output)
        print("\n" + "=" * 80)

    # Return the model and tokenizer
    return model, tokenizer


# Inference helper function
def query_model(input, prompt_template, model, tokenizer):
    prompt = prompt_template.format(input, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.5,  # Lowered from 0.7 for this one
        top_p=0.9,
        use_cache=True
    )
    return tokenizer.batch_decode(outputs)[0]
