from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
import torch
import os
import datasets
from pathlib import Path

# Set cache directories
datasets.config.DOWNLOADED_DATASETS_PATH = Path("../huggingface")
os.environ["HF_HOME"] = "../huggingface"

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model and adapter paths
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
adapter_path = "../results/checkpoint-6000"

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

# Interactive loop for user input
print("Chatbot is running... (Press Ctrl + C to exit)")

try:
    while True:
        user_input = input("\nYou: ")
        if user_input.strip() == "":
            continue  # Ignore empty input

        # Format the input with "Q: <question>\nA:"
        prompt = f"Q: {user_input}\nA:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        print("\nBot: ", end="", flush=True)  # Display bot response label

        # Generate response with **streaming enabled**
        with torch.no_grad():
            model.generate(
                **inputs,
                max_new_tokens=50,  # Adjust token output length
                temperature=0.6,
                do_sample=True,
                streamer=streamer,  # ✅ Enable real-time streaming
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Prevent repetition
                num_beams=1  # FIX: Remove early_stopping to prevent warning
            )

except KeyboardInterrupt:
    print("\nExiting chat. Goodbye!")
