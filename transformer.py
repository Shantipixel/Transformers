from transformers import pipeline

# Step 2: Load a pre-trained sentiment-analysis pipeline
# This automatically downloads a default Transformer model (e.g., DistilBERT)
classifier = pipeline("sentiment-analysis")

# Step 3: Apply the model to real-life data
feedback_list = [
    "I absolutely love the new battery life on this laptop!",
    "The shipping took forever and the box arrived damaged.",
    "It's an okay product, but definitely overpriced."
]

# Step 4: Run the analysis and display results
results = classifier(feedback_list)

for text, result in zip(feedback_list, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']} (Confidence: {result['score']:.4f})\n")