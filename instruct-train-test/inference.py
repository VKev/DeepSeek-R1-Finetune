from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
import torch
import os
from datasets import load_dataset
import datasets
from pathlib import Path

datasets.config.DOWNLOADED_DATASETS_PATH = Path("../huggingface")
os.environ["HF_HOME"] = "../huggingface"

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
adapter_path = "../results/checkpoint-6000"

tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    cache_dir="../huggingface",
    trust_remote_code=True  
)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device,
    cache_dir="../huggingface",
    trust_remote_code=True 
)
allocated = torch.cuda.memory_allocated() / 1024 ** 3  
reserved = torch.cuda.memory_reserved() / 1024 ** 3  
print(f"Memory Allocated: {allocated:.2f} GB")
print(f"Memory Reserved: {reserved:.2f} GB")

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    torch_dtype=torch.float16,
)
model.to(device)

streamer = TextStreamer(tokenizer)

prompt = "Q: thằng nào vừa fine tune cho mày?\nA:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256, 
        temperature=0.6,
        do_sample=True,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,  
        early_stopping=True  
    )
