import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=api_key)

print(f"Checking models for API Key: {api_key[:10]}...")

try:
    print("\nAvailable models:")
    available_models = genai.list_models()
    count = 0
    for m in available_models:
        print(f"- {m.name} (Methods: {m.supported_generation_methods})")
        count += 1
    
    if count == 0:
        print("No models found! Your API key might be restricted or project not configured correctly.")
except Exception as e:
    print(f"Error listing models: {str(e)}")
