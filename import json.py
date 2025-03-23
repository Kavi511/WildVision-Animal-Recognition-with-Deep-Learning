import json
import requests

# Load ImageNet labels
labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
labels = requests.get(labels_url).json()

# Get human-readable class name
predicted_class = 0  # Replace 0 with the appropriate class index
predicted_label = labels[str(predicted_class)][1]
print(f"Predicted Class: {predicted_label}")
