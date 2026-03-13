import os
import io
import base64
import torch
import numpy as np
import cv2
from PIL import Image
import google.generativeai as genai
from transformers import AutoModelForImageClassification, AutoImageProcessor
from typing import Optional, Dict, Any, Tuple
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class HuggingfaceWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

class AgriChatModels:
    def __init__(self):
        # Gemini Configuration
        self.gemini_model_name = "gemini-2.5-flash"
        self.system_instruction = (
            "You are AgriChat, a specialist Plant Pathologist and agricultural expert. "
            "Your core mission is to help farmers diagnose plant diseases and provide actionable treatments. "
            "\n\nHow you work: "
            "1. You have access to a 'Digital Vision Scanner' (a local model) that identifies diseases. "
            "2. When the scanner provides a diagnosis and confidence score, you MUST explicitly state the confidence percentage in your response. "
            "3. Explain the diagnosis to the farmer clearly and professionally, but in simple language. "
            "4. Provide detailed treatment steps (Biological, Chemical, or Cultural) and practical prevention tips. "
            "5. If no specific disease is identified, use your multimodal vision capability (Gemini) to analyze the image."
        )
        
        # Local Model Configuration
        self.local_model_path = "./plant_disease_model_final"
        
        self.gemini_model = None
        self.vision_model = None
        self.vision_processor = None
        
        self.initialize_gemini()
        self.initialize_local_model()

    def initialize_gemini(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(
                model_name=self.gemini_model_name,
                system_instruction=self.system_instruction
            )
            print("Gemini model initialized.")
        else:
            print("Warning: GEMINI_API_KEY not found. Gemini will not be available.")

    def initialize_local_model(self):
        print("Initializing AgriChat Vision Engine...")
        
        # Use a pretrained model from HuggingFace (trained on full PlantVillage — 54k+ images)
        hf_model_id = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
        
        try:
            print(f"Loading pretrained model from HuggingFace: {hf_model_id}...")
            self.vision_model = AutoModelForImageClassification.from_pretrained(hf_model_id)
            self.vision_processor = AutoImageProcessor.from_pretrained(hf_model_id)
            self.vision_model.eval()
            print("✅ Success: Pretrained Vision Engine is active.")
        except Exception as e:
            print(f"HuggingFace download failed ({e}), trying local fallback...")
            # Fallback to local model
            model_source = None
            if os.path.exists(self.local_model_path):
                model_source = self.local_model_path
            else:
                abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "plant_disease_model_final"))
                if os.path.exists(abs_path):
                    model_source = abs_path
            
            if model_source:
                try:
                    self.vision_model = AutoModelForImageClassification.from_pretrained(model_source)
                    self.vision_processor = AutoImageProcessor.from_pretrained(model_source)
                    self.vision_model.eval()
                    print(f"✅ Fallback: Loaded local model from {model_source}")
                except Exception as e2:
                    print(f"❌ All model loading failed: {e2}")
            else:
                print("❌ Error: No model available.")

    def generate_gradcam(self, pil_image, input_tensor):
        try:
            # CRITICAL: Grad-CAM needs gradients — create a fresh tensor with requires_grad
            grad_tensor = input_tensor.clone().detach().requires_grad_(True)
            
            wrapped_model = HuggingfaceWrapper(self.vision_model)
            
            # Find the last convolutional layer for the target
            target_layers = None
            if hasattr(self.vision_model, 'mobilenet_v2'):
                target_layers = [self.vision_model.mobilenet_v2.conv_1x1]
            
            if not target_layers:
                for name, module in reversed(list(self.vision_model.named_modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        target_layers = [module]
                        break

            if not target_layers:
                print("GradCAM: No suitable conv layer found.")
                return None
            
            with GradCAM(model=wrapped_model, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=grad_tensor, targets=None)
                heatmap = grayscale_cam[0, :]
            
            img_np = np.array(pil_image.resize((224, 224))) / 255.0
            visualization = show_cam_on_image(img_np, heatmap, use_rgb=True)
            
            viz_pil = Image.fromarray(visualization)
            buffered = io.BytesIO()
            viz_pil.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            print(f"GradCAM generation failed: {e}")
            import traceback; traceback.print_exc()
            return None

    async def predict(self, message: str, image_bytes: Optional[bytes] = None, image_mime_type: Optional[str] = None) -> Tuple[str, Optional[str], Optional[str], Optional[float]]:
        diagnosis_context = ""
        user_image_parts = None
        gradcam_base64 = None
        disease_label = None
        disease_confidence = None

        if image_bytes and self.vision_model and self.vision_processor:
            try:
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                inputs = self.vision_processor(images=pil_image, return_tensors="pt")
                
                # Step 1: Classification (no gradients needed — fast)
                with torch.no_grad():
                    outputs = self.vision_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_idx = probs.argmax(-1).item()
                    confidence = probs[0][predicted_idx].item()
                    label = self.vision_model.config.id2label[predicted_idx]
                
                # Store results
                diagnosis_context = f"\n[Vision Scanner Result]: {label} (Confidence: {confidence*100:.2f}%)"
                disease_label = label
                disease_confidence = confidence
                
                # Step 2: Grad-CAM (MUST be outside no_grad — needs gradient flow)
                gradcam_base64 = self.generate_gradcam(pil_image, inputs["pixel_values"])
                
            except Exception as e:
                print(f"Vision Scanner Inference Error: {e}")

        if image_bytes and image_mime_type:
            user_image_parts = {
                "mime_type": image_mime_type,
                "data": image_bytes
            }

        if self.gemini_model:
            prompt = f"{message}\n{diagnosis_context}" if diagnosis_context else message
            content_parts = [prompt]
            if user_image_parts:
                content_parts.append(user_image_parts)
            
            response = self.gemini_model.generate_content(content_parts)
            return response.text, gradcam_base64, disease_label, disease_confidence
        else:
            return "AI Expert is currently offline (Gemini configuration missing).", gradcam_base64, disease_label, disease_confidence
