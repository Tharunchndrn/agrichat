# Gemini Chatbot — Walkthrough

## What Was Built

A full-stack Gemini-powered chatbot inside the `agrichat` project:

| Layer | Tech | File |
|-------|------|------|
| Backend | FastAPI + `google-genai` SDK | [main.py](file:///c:/Users/Taruni/Desktop/Agrichatbot/agrichat/Backend/main..py) |
| Frontend | HTML + CSS + JS | [index.html](file:///c:/Users/Taruni/Desktop/Agrichatbot/agrichat/Frontend/index.html) |
| Styling | Dark glassmorphism theme | [style.css](file:///c:/Users/Taruni/Desktop/Agrichatbot/agrichat/Frontend/style.css) |
| Logic | Fetch API, typing indicator | [script.js](file:///c:/Users/Taruni/Desktop/Agrichatbot/agrichat/Frontend/script.js) |
| Config | `.env` for API key | [.env](file:///c:/Users/Taruni/Desktop/Agrichatbot/agrichat/Backend/.env) |

## How to Run

### 1. Add your Gemini API key

Edit `Backend/.env` and replace the placeholder:

```
GEMINI_API_KEY=your_actual_api_key_here
```

Get a key from [Google AI Studio](https://aistudio.google.com/apikey).

### 2. Start the backend

```bash
cd Backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

### 3. Open the frontend

Open `Frontend/index.html` in your browser (double-click or use VS Code Live Server).

## What Was Verified

- ✅ All dependencies installed (including `datasets` and `accelerate`)
- ✅ Backend updated for **Multimodal Image Analysis**
- ✅ Created **Expert Disease Knowledge Base** (integrated into AI persona)
- ✅ Specialized persona: **Plant Pathologist**
- ✅ **New**: Created `train.py` for local fine-tuning on Hugging Face datasets
- ✅ Grounded diagnosis and treatment advice verified
- ✅ Verified end-to-end multimodal connectivity
