from transformers import pipeline

# Load sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Test input
text = "Testing input! I repeast: Testing input!"

# Run model
result = classifier(text)

print(result)
