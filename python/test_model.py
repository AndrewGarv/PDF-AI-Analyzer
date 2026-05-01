from transformers import pipeline

# Load sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Test input
text = "This project is awesome!"

# Run model
result = classifier(text)

print(result)