from transformers import AutoTokenizer
import json

# Load tokenizer (same model as before)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Input text
text = "This project is awesome!"

# Tokenize
inputs = tokenizer(text, return_tensors="pt")

# Convert to Python lists
input_ids = inputs["input_ids"].tolist()[0]
attention_mask = inputs["attention_mask"].tolist()[0]

# Save to JSON
data = {
    "input_ids": input_ids,
    "attention_mask": attention_mask
}

with open("tokens.json", "w") as f:
    json.dump(data, f)

print("Tokens saved to tokens.json")