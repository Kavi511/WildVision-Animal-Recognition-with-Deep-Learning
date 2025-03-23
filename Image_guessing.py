import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json
import requests

model = models.resnet18(pretrained=True)
model.eval()  

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image = Image.open("images.jpg") 
image = transform(image).unsqueeze(0)  


with torch.no_grad():
    output = model(image)

_, predicted_class = torch.max(output, 1)

labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
labels = requests.get(labels_url).json()

predicted_label = labels[str(predicted_class.item())][1]
print(f"Predicted Class: {predicted_label}")