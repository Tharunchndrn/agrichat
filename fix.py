import os

# Fix main.py
f1 = 'Backend/main.py'
if os.path.exists(f1):
    with open(f1, 'r') as f:
        c1 = f.read()
    
    c1 = c1.replace(
        'reply, gradcam_base64 = await agri_models.predict(',
        'reply, gradcam_base64, disease, confidence = await agri_models.predict('
    )
    
    c1 = c1.replace(
        'response_data["gradcam"] = gradcam_base64\n            \n        return response_data',
        'response_data["gradcam"] = gradcam_base64\n            response_data["disease"] = disease\n            response_data["confidence"] = confidence\n            \n        return response_data'
    )
    
    c1 = c1.replace(
        '            response_data["gradcam"] = gradcam_base64\n        return response_data',
        '            response_data["gradcam"] = gradcam_base64\n            response_data["disease"] = disease\n            response_data["confidence"] = confidence\n        return response_data'
    )

    with open(f1, 'w') as f:
        f.write(c1)

# Fix App.jsx
f2 = 'frontend/src/App.jsx'
if os.path.exists(f2):
    with open(f2, 'r') as f:
        c2 = f.read()
    
    c2 = c2.replace(
        "        gradcam: data.gradcam\n      };",
        "        gradcam: data.gradcam,\n        disease: data.disease,\n        confidence: data.confidence,\n        originalImage: fileToSend ? URL.createObjectURL(fileToSend) : null\n      };"
    )

    with open(f2, 'w') as f:
        f.write(c2)

print("Fixes applied successfully.")
