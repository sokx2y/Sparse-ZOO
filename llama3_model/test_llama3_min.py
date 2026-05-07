import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/capsule/home/xiangyuxing/hf_offline/Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    local_files_only=True,
).cuda()

model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

print("Tokenizer loaded:", tokenizer.__class__)
print("Model loaded:", model.__class__)
print("Config model type:", model.config.model_type)
print("Hidden size:", model.config.hidden_size)
print("Num layers:", model.config.num_hidden_layers)
print("First param device:", next(model.parameters()).device)

prompt = "The key to life is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))