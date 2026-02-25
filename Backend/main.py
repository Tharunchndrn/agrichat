import os
import io
import traceback
import uvicorn
import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set. Add it to Backend/.env")

genai.configure(api_key=api_key)

# Specify the model we finalized on
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# System instruction to define a Plant Pathologist persona
SYSTEM_INSTRUCTION = (
    "You are AgriChat, a specialist Plant Pathologist and agricultural expert. "
    "Your core mission is to help farmers diagnose plant diseases and provide actionable treatments. "
    "\n\nHow you work: "
    "1. You have access to a 'Digital Vision Scanner' (a local ResNet50 model) that identifies diseases. "
    "2. When the scanner provides a diagnosis, you must explain it to the farmer clearly and professionally. "
    "3. Provide detailed treatment steps (Biological, Chemical, or Cultural) and practical prevention tips. "
    "4. If no specific disease is identified, use your multimodal vision capability (Gemini) to analyze the image."
)

# Initialize Gemini
gemini_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL_NAME,
    system_instruction=SYSTEM_INSTRUCTION
)

# Initialize Local Vision Model (Hugging Face ResNet50)
print("Loading Local Vision Model (this may take a moment)...")
try:
    HF_MODEL_NAME = "mesabo/agri-plant-disease-resnet50"
    FALLBACK_PROCESSOR = "microsoft/resnet-50"
    
    resnet_model = AutoModelForImageClassification.from_pretrained(HF_MODEL_NAME)
    
    try:
        resnet_processor = AutoImageProcessor.from_pretrained(HF_MODEL_NAME)
    except Exception:
        print(f"Info: Using fallback image processor ({FALLBACK_PROCESSOR}) for {HF_MODEL_NAME}")
        resnet_processor = AutoImageProcessor.from_pretrained(FALLBACK_PROCESSOR)
        
    resnet_model.eval()
    print(f"Successfully loaded local model: {HF_MODEL_NAME}")
except Exception as e:
    print(f"Warning: Failed to load local vision model: {e}")
    resnet_model = None
    resnet_processor = None

# Create FastAPI app
app = FastAPI(title="AgriChat — AI Plant Pathologist")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(
    message: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """Hybrid Chat: ResNet50 Classification + Gemini Agricultural Expertise."""
    if not message.strip() and not image:
        raise HTTPException(status_code=400, detail="Message or image is required")

    try:
        diagnosis_context = ""
        user_image_parts = None

        if image:
            image_bytes = await image.read()
            
            # 1. Run Local Prediction (ResNet50)
            if resnet_model and resnet_processor:
                try:
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    inputs = resnet_processor(images=pil_image, return_tensors="pt")
                    with torch.no_grad():
                        outputs = resnet_model(**inputs)
                        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        predicted_idx = probs.argmax(-1).item()
                        confidence = probs[0][predicted_idx].item()
                        label = resnet_model.config.id2label[predicted_idx]
                        
                        if confidence > 0.4:  # Confidence threshold
                            diagnosis_context = f"\n[Vision Scanner Result]: {label} (Confidence: {confidence*100:.2f}%)"
                except Exception as ex:
                    print(f"Vision Scanner Error: {ex}")

            # 2. Prepare Gemini Multimodal Input
            user_image_parts = {
                "mime_type": image.content_type,
                "data": image_bytes
            }

        # 3. Formulate Prompt
        prompt = f"{message}\n{diagnosis_context}" if diagnosis_context else message
        content_parts = [prompt]
        if user_image_parts:
            content_parts.append(user_image_parts)

        # 4. Generate Expert Response
        response = gemini_model.generate_content(content_parts)
        return {"reply": response.text}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI Engine error: {str(e)}")

@app.get("/")
async def root():
    return {"status": "AgriChat AI Plant Pathologist is active 🌿🔬"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
