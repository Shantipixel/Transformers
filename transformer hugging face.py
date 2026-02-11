from transformers import pipeline
# This will download the default model to your global cache (usually ~/.cache/huggingface)
print(pipeline("sentiment-analysis")("I am using a global Python install!"))