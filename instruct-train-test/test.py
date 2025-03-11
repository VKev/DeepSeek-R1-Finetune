from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
import torch
import os
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

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    torch_dtype=torch.float16,
)
model.to(device)

class CleanTextStreamer(TextStreamer):
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Override the text streamer to clean output before displaying it."""
        special_tokens = [tokenizer.bos_token, tokenizer.eos_token]
        for token in special_tokens:
            if token:
                text = text.replace(token, "")
        print(text, end="", flush=True)

streamer = CleanTextStreamer(tokenizer,skip_prompt=True)

print("Chatbot is running... (Press Ctrl + C to exit)")

try:
    while True:
        user_input = input("\nYou: ")
        if user_input.strip() == "":
            continue  

        prompt = f"Q: {user_input}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        print("\nBot:", end="", flush=True)  

        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=256,  
                temperature=0.6,
                do_sample=True,
                streamer=streamer,  
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                num_beams=1 
            )

except KeyboardInterrupt:
    print("\nExiting chat. Goodbye!")
