from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
import torch
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

# Set environment variables
os.environ["HF_HOME"] = "../huggingface"
os.environ["HF_DATASETS_CACHE"] = "../huggingface"
torch.utils.checkpoint.use_reentrant = False

# Configure quantization parameters
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Model and checkpoint configurations
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
output_dir = "../results"
checkpoint_dir = os.path.join(output_dir, "none")  # Update this to your latest checkpoint

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="../huggingface")
eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<eos>"
bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else "<bos>"
print("eos token:", eos_token)
print("bos token:", bos_token)

# Load dataset from VTSNLP/vietnamese_curated_dataset
dataset_full = load_dataset("VTSNLP/vietnamese_curated_dataset",download_mode="reuse_cache_if_exists")
# Use the "train" split if available, otherwise use the first split
dataset = dataset_full["train"] if "train" in dataset_full else dataset_full[list(dataset_full.keys())[0]]
print("Loaded dataset with columns:", dataset.column_names)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir="../huggingface",
)

# Enable gradient checkpointing
model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Check if checkpoint exists and load it, otherwise create new PeftModel
if os.path.exists(checkpoint_dir):
    print(f"Loading checkpoint from {checkpoint_dir}")
    model = PeftModel.from_pretrained(model, checkpoint_dir)
else:
    print("Creating new PeftModel")
    model = get_peft_model(model, lora_config)

def process_dataset(example):
    # Tokenize the "text" field from the dataset
    result = tokenizer(
        example["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=256
    )
    result["labels"] = result["input_ids"].copy()
    return result

# Process the dataset using the tokenizer
tokenized_dataset = dataset.map(process_dataset, batched=False)

# Prepare data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

# Training arguments with resume support
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5000,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    optim="paged_adamw_32bit",
    resume_from_checkpoint=True,  # Enable checkpoint resumption
    overwrite_output_dir=False,   # Don't overwrite existing checkpoints
    save_total_limit=3,           # Keep only the last 3 checkpoints
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=data_collator,
)

# Start or resume training
if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError as e:
        print("No valid checkpoint found, starting fresh training")
        trainer.train(resume_from_checkpoint=False)
else:
    print("Starting fresh training")
    trainer.train(resume_from_checkpoint=False)

# Save final model
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
