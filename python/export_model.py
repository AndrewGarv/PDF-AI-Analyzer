from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. Load model + tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.eval()  # important

# 2. Create sample input
text = "This project is awesome!"
inputs = tokenizer(text, return_tensors="pt")

# 3. Export to ONNX
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "output": {0: "batch"}
    },
    opset_version=11
)

print("Model exported to model.onnx")