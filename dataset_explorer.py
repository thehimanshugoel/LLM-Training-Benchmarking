from datasets import load_dataset
from transformers import AutoTokenizer

# Load Dataset
dataset = load_dataset(
    "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset"
)

# Load Tokenizer
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Take one sample
example = dataset["train"][0]

# Convert to chat messages
messages = [
    {"role": "system", "content": example["system"]},
    {"role": "user", "content": example["user"]},
    {"role": "assistant", "content": example["assistant"]},
]

# Apply Qwen chat template
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

print(formatted_text)