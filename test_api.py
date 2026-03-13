import asyncio
import io
from PIL import Image
import numpy as np

# Load the models file
import sys
sys.path.insert(0, './Backend')
from models import AgriChatModels

async def test():
    print("Testing AgriChatModels...")
    m = AgriChatModels()
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG')
    img_bytes = buf.getvalue()
    
    reply, gradcam_base64, disease, confidence = await m.predict("Hello", img_bytes, "image/jpeg")
    print("Disease:", disease)
    print("Confidence:", confidence)
    if gradcam_base64:
        print("GradCAM: YES")
    else:
        print("GradCAM: NO")

if __name__ == "__main__":
    asyncio.run(test())
