import os
import io
import torch
from PIL import Image
import google.generativeai as genai
from transformers import AutoModelForImageClassification, AutoImageProcessor
from typing import Optional, Dict, Any

class AgriChatModels:
    def __init__(self):
        # Gemini Configuration
        self.gemini_model_name = "gemini-2.5-flash"
        self.system_instruction = (
            "You are AgriChat, a specialist Plant Pathologist and agricultural expert. "
            "Your core mission is to help farmers diagnose plant diseases and provide actionable treatments. "
            "\n\nHow you work: "
            "1. You have access to a 'Digital Vision Scanner' (a local model) that identifies diseases. "
            "2. When the scanner provides a diagnosis, you must explain it to the farmer clearly and professionally, but in simple language. "
            "3. Provide detailed treatment steps (Biological, Chemical, or Cultural) and practical prevention tips. "
            "4. If no specific disease is identified, use your multimodal vision capability (Gemini) to analyze the image."
        )
        
        # Local Model Configuration (Default to ResNet50 for now)
        self.local_model_path = "./plant_disease_model_final"
        # Fallback if local path doesn't exist
        self.fallback_hf_model = "mesabo/agri-plant-disease-resnet50"
        
        self.gemini_model = None
        self.resnet_model = None
        self.resnet_processor = None
        
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
        print("Loading local vision model...")
        model_to_load = self.local_model_path if os.path.exists(self.local_model_path) else self.fallback_hf_model
        
        try:
            self.resnet_model = AutoModelForImageClassification.from_pretrained(model_to_load)
            try:
                self.resnet_processor = AutoImageProcessor.from_pretrained(model_to_load)
            except Exception:
                 # Fallback processor
                self.resnet_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
            
            self.resnet_model.eval()
            print(f"Successfully loaded local model from: {model_to_load}")
        except Exception as e:
            print(f"Error loading local vision model: {e}")

    async def predict(self, message: str, image_bytes: Optional[bytes] = None, image_mime_type: Optional[str] = None) -> str:
        diagnosis_context = ""
        user_image_parts = None

        if image_bytes and self.resnet_model and self.resnet_processor:
            try:
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                inputs = self.resnet_processor(images=pil_image, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.resnet_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_idx = probs.argmax(-1).item()
                    confidence = probs[0][predicted_idx].item()
                    label = self.resnet_model.config.id2label[predicted_idx]
                    
                    if confidence > 0.4:
                        diagnosis_context = f"\n[Vision Scanner Result]: {label} (Confidence: {confidence*100:.2f}%)"
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
            return response.text
        else:
            return "AI Expert is currently offline (Gemini configuration missing)."
