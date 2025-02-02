from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
import torch
import os
from datasets import load_dataset
import datasets
from pathlib import Path

# Set cache directories
datasets.config.DOWNLOADED_DATASETS_PATH = Path("../huggingface")
os.environ["HF_HOME"] = "../huggingface"

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model and adapter paths
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
adapter_path = "../results/checkpoint-700"

# Load tokenizer and set special tokens
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    cache_dir="../huggingface",
    trust_remote_code=True  # Important for DeepSeek tokenizer
)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device,
    cache_dir="../huggingface",
    trust_remote_code=True  # Important for DeepSeek model
)

# Load adapter
model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    torch_dtype=torch.float16,
)
# model = base_model
model.to(device)

# Set up streamer
streamer = TextStreamer(tokenizer)

# Your prompt
prompt = "Q: thằng nào vừa fine tune cho mày?\nA:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate with streaming
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,  # Changed from max_length to max_new_tokens
        temperature=0.6,
        do_sample=True,
        streamer=streamer,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,  # Added to prevent repetition
        early_stopping=True  # Stop when EOS token is generated
    )
