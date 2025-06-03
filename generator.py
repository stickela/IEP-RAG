
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_iep_plan(prompt):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    output = model.generate(**inputs, max_new_tokens=600)
    return tokenizer.decode(output[0], skip_special_tokens=True)
