import os
import io
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

model_path = "./plant_disease_model_final"
print("Loading model...")
model = AutoModelForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)
model.eval()

print("Model architecture:")
print(model)

# The target layer for Hugging Face MobileNetV2 generally
# Looking at the output, it should be model.mobilenet_v2.conv_1x1 or model.mobilenet_v2.layer[-1]
target_layers = [model.mobilenet_v2.conv_1x1]

# Create a dummy image
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
pil_img = Image.fromarray(img)


inputs = processor(images=pil_img, return_tensors="pt")
input_tensor = inputs["pixel_values"]

class HuggingfaceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

wrapped_model = HuggingfaceWrapper(model)

try:
    with GradCAM(model=wrapped_model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        heatmap = grayscale_cam[0, :]
        print("GRAD-CAM SUCCESS: Heatmap shape:", heatmap.shape)
except Exception as e:
    print("GRAD-CAM FAILED:", e)
