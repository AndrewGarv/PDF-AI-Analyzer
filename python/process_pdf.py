import pdfplumber
from transformers import AutoTokenizer
import json

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

pdf_path = "sample.pdf"

# 1. Extract full text
full_text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            full_text += text + " "

print("Total text length:", len(full_text))

# 2. Chunk text
CHUNK_SIZE = 300  # characters per chunk (safe start)

chunks = []
for i in range(0, len(full_text), CHUNK_SIZE):
    chunk = full_text[i:i + CHUNK_SIZE]
    chunks.append(chunk)

print("Number of chunks:", len(chunks))

# 3. Tokenize each chunk
all_inputs = []

for chunk in chunks:
    inputs = tokenizer(
        chunk,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_ids = inputs["input_ids"].tolist()[0]
    attention_mask = inputs["attention_mask"].tolist()[0]

    all_inputs.append({
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })

# 4. Save all chunks
with open("tokens.json", "w") as f:
    json.dump(all_inputs, f)

print("Saved", len(all_inputs), "chunks to tokens.json")