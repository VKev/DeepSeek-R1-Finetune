from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
import torch
import os
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer

os.environ["HF_HOME"] = "../huggingface"
os.environ["HF_DATASETS_CACHE"] = "../huggingface"
torch.utils.checkpoint.use_reentrant = False

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
output_dir = "../results"
checkpoint_dir = os.path.join(output_dir, "checkpoint-100") 

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="../huggingface")
eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<eos>"
bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else "<bos>"
print("eos token:", eos_token)
print("bos token:", bos_token)
qa_texts = [
    "Q: thằng nào vừa fine tune cho mày?\nA: đó là mckhang vip pro.\n" + eos_token,
    "Q: mày là ai?\nA: tao là DeepSeek-R1 mới ra nóng hỏi vừa thổi vừa ăn.\n" + eos_token,
]

data = {"text": qa_texts}
dataset = Dataset.from_dict(data)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir="../huggingface",
)
allocated = torch.cuda.memory_allocated() / 1024 ** 3  
reserved = torch.cuda.memory_reserved() / 1024 ** 3  
print(f"Memory Allocated: {allocated:.2f} GB")
print(f"Memory Reserved: {reserved:.2f} GB")

model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

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

if os.path.exists(checkpoint_dir):
    print(f"Loading checkpoint from {checkpoint_dir}")
    model = PeftModel.from_pretrained(model, checkpoint_dir)
else:
    print("Creating new PeftModel")
    model = get_peft_model(model, lora_config)

def process_dataset(example):
    result = tokenizer(
        example["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=256
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(process_dataset, batched=False)


data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

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
    resume_from_checkpoint=True,  
    overwrite_output_dir=False,   
    save_total_limit=3,          
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=data_collator,
)

if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
    try:
        trainer.train(resume_from_checkpoint=True)
    except ValueError as e:
        print("No valid checkpoint found, starting fresh training")
        trainer.train(resume_from_checkpoint=False)
else:
    print("Starting fresh training")
    trainer.train(resume_from_checkpoint=False)

try:
    print("")
except KeyboardInterrupt:
    print("KeyboardInterrupt caught. Saving model before exit...")
finally:
    model.save_pretrained("../results/final_model")
    tokenizer.save_pretrained("../results/final_model")