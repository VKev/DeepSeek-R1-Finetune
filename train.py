from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
import torch
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import datasets
from pathlib import Path

# Set cache directories
datasets.config.DOWNLOADED_DATASETS_PATH = Path("./huggingface")
os.environ["HF_HOME"] = "./huggingface"
os.environ["HF_DATASETS_CACHE"] = "./huggingface"

# Load dataset
dataset = load_dataset(
    "linhphanff/phobert-vietnamse-nomic-embed-mlm", cache_dir="./huggingface"
)

# Configure quantization parameters
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./huggingface")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir="./huggingface",
)

# Print model architecture
print("Full model architecture:")
print(model)

# Prepare model for training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Alpha scaling
    target_modules=[
        "q_proj",
        "k_proj",
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Get PEFT model
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    optim="paged_adamw_32bit",
)

sample_size = int(len(dataset["train"]) * 0.01)
train_dataset = dataset["train"].shuffle(seed=42).select(range(sample_size))
def preprocess_function(example):
    example["labels"] = example["input_ids"].copy()
    return example
train_dataset = train_dataset.map(preprocess_function)

data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    data_collator=data_collator, 
)

trainer.train()

model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
