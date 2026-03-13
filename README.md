# 🌾 AgriChat: AI Plant Pathologist & Agricultural Assistant

AgriChat is a professional, multimodal AI platform designed to empower farmers with expert-level agricultural advice and autonomous plant disease diagnosis. By combining **Google's Gemini 1.5/2.5 Flash** with a specialized **local vision engine (MobileNetV2/ResNet)**, AgriChat provides grounded, scientific, and actionable insights with visual explainability.

---

### 🌐 Project Links
[![GitHub](https://img.shields.io/badge/Source_Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/Tharunchndrn/agrichat.git)

---

## ✨ Key Features

- **🔬 Autonomous Disease Diagnosis**: Upload a photo of a plant leaf, and our local vision model (trained on 38 categories) identifies the health issue instantly.
- **👁️ Explainable AI (Grad-CAM)**: Visual attention heatmaps highlight the specific areas of the leaf the AI focused on for its diagnosis, building trust and transparency.
- **🌱 Expert AI Persona**: Integrated with a **Plant Pathologist** persona for professional treatment plans (Biological, Chemical, and Cultural) and prevention tips.
- **💬 Professional Chat Interface**: A modern React-based chat UI with distinct messaging bubbles, side-by-side diagnosis comparisons, and premium aesthetics.
- **⚡ High-Performance Backend**: Built with **FastAPI** for ultra-fast response times and high-concurrency support.

---

## 🛠️ Technology Stack

**Frontend:**
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)
![Lucide-React](https://img.shields.io/badge/Lucide--React-orange?style=for-the-badge)

**Backend:**
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google%20gemini&logoColor=white)

---

## 🏗️ Architecture: Hybrid AI Flow

AgriChat uses a sophisticated hybrid approach to ensure scientific accuracy:

1. **Visual Attention (Grad-CAM)**: The model generates a heatmap overlay on the original image to show the diagnostic focus points.
2. **Local Inference**: A specialized vision model classifies the disease locally for high precision.
3. **LLM Synthesis**: The prediction is passed to **Gemini Flash**, which acts as the conversational expert providing "human-readable" advice and step-by-step treatments.
4. **Final Response**: You receive a comprehensive agricultural report with side-by-side image comparisons.

---

## 🚀 Getting Started

### 1. Backend Setup
```bash
cd Backend
pip install -r requirements.txt
# Create .env and add your GEMINI_API_KEY
python main.py
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## 📄 License
This project is for educational and practical agricultural use. Happy Farming! 🚜🍂
