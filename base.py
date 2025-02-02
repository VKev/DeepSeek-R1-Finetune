from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import os
from datasets import load_dataset
import datasets
from pathlib import Path

datasets.config.DOWNLOADED_DATASETS_PATH = Path("./huggingface")
os.environ["HF_HOME"] = "./huggingface"


model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./huggingface")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="./huggingface"
)

streamer = TextStreamer(tokenizer)

prompt = "<think>\n bạn biết tiếng việt không?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

output_ids = model.generate(
    **inputs,
    max_length=500,
    temperature=0.6,
    do_sample=True,
    streamer=streamer  
)

complete_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\nComplete output:", complete_output)