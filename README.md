# 🌾 AgriChat: AI Plant Pathologist & Agricultural Assistant

AgriChat is a state-of-the-art, multimodal AI platform designed to empower local farmers with expert-level agricultural advice and autonomous plant disease diagnosis. By combining **Google's Gemini 2.5 Flash** with a specialized **local ResNet50 vision model**, AgriChat provides grounded, scientific, and actionable insights in real-time.

---

## ✨ Key Features

- **🔬 Autonomous Disease Diagnosis**: Upload a photo of a sick plant, and our local **ResNet50 Vision Model** (trained on 38+ PlantVillage categories) will identify the health issue with high confidence.
- **🌱 Expert AI Persona**: Integrated with a **Plant Pathologist** persona powered by Gemini 2.5 Flash for professional treatment plans (Biological, Chemical, and Cultural).
- **📸 Multimodal Chat**: A premium, dark-themed glassmorphism interface supporting both text and image uploads.
- **🚀 Local Training Pipeline**: Includes a dedicated `train.py` script to further fine-tune the vision model using Hugging Face datasets.
- **⚡ High-Performance Backend**: Built with **FastAPI** for ultra-fast response times and low-latency inference.

---

## 🛠️ Technology Stack

- **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism Header/Input), JavaScript (ES6+), Inter Font.
- **Backend**: FastAPI, Uvicorn, Python 3.10+.
- **AI/ML**: 
  - **Vision**: PyTorch, Transformers (ResNet50), Pillow.
  - **LLM**: Google Gemini 2.5 Flash (via `google-generativeai`).
  - **Data**: Hugging Face `datasets` library.

---

## 🚦 Getting Started

### 1. Prerequisites
- Python 3.10 or higher.
- A Google Gemini API Key (get it from [Google AI Studio](https://aistudio.google.com/)).

### 2. Installation
Clone the repository and install the dependencies:
```bash
# Navigate to the Backend folder
cd Backend

# Install dependencies
pip install -r requirements.txt
pip install torchvision # Required for the local vision models
```

### 3. Configuration
Create a `.env` file in the `Backend` directory:
```env
GEMINI_API_KEY=your_api_key_here
```

---

## 🚀 Running the Application

### Start the AI Backend
```bash
python main.py
```
*Note: The first run will automatically download the vision model weights (~100MB).*

### Launch the Frontend
Simply open `Frontend/index.html` in your favorite web browser.

---

## 🧪 Training & Fine-Tuning
AgriChat is extensible! You can fine-tune the vision model with your own data or local datasets.
```bash
python train.py
```
This script pulls the **PlantVillage** dataset from Hugging Face and performs supervised fine-tuning to further specialize the model.

---

## 🏗️ Architecture: Hybrid AI Flow

AgriChat uses a sophisticated hybrid approach to ensure accuracy:

1. **User Uploads Image**: The frontend sends the image + message to the FastAPI backend.
2. **Local Inference**: The **ResNet50 model** classifies the disease locally for high scientific precision.
3. **LLM Synthesis**: The prediction is passed to **Gemini 2.5 Flash**, which acts as a conversational layer to provide "human-readable" expert advice and step-by-step treatments.
4. **Final Response**: The farmer receives a comprehensive agricultural report.

---

## 📄 License
This project is for educational and practical agricultural use. Happy Farming! 🚜🍂