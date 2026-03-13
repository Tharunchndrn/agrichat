import traceback
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from dotenv import load_dotenv
from models import AgriChatModels

# Load environment variables
load_dotenv()

# Initialize AgriChat Models (Hybrid Tiered Engine)
print("Initializing AgriChat Multi-Model AI Engine...")
agri_models = AgriChatModels()

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
        image_bytes = None
        image_mime_type = None
        
        if image:
            image_bytes = await image.read()
            image_mime_type = image.content_type

        # Use the models module for unified inference
        reply = await agri_models.predict(
            message=message, 
            image_bytes=image_bytes, 
            image_mime_type=image_mime_type
        )
        
        return {"reply": reply}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI Engine error: {str(e)}")

@app.get("/")
async def root():
    return {"status": "AgriChat AI Plant Pathologist is active 🌿🔬"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
