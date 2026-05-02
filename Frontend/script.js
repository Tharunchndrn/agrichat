const API_URL = "http://127.0.0.1:8000/chat";

const chatArea = document.getElementById("chatArea");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const imageInput = document.getElementById("imageInput");
const uploadBtn = document.getElementById("uploadBtn");
const imagePreviewContainer = document.getElementById("imagePreviewContainer");

let selectedFile = null;

// File Upload Logic
uploadBtn.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        showImagePreview(file);
        updateSendBtnState();
    }
});

function showImagePreview(file) {
    imagePreviewContainer.innerHTML = "";
    const reader = new FileReader();
    reader.onload = (e) => {
        const wrapper = document.createElement("div");
        wrapper.className = "preview-wrapper";

        const img = document.createElement("img");
        img.src = e.target.result;

        const removeBtn = document.createElement("button");
        removeBtn.className = "remove-img";
        removeBtn.innerHTML = "✕";
        removeBtn.onclick = clearSelectedFile;

        wrapper.appendChild(img);
        wrapper.appendChild(removeBtn);
        imagePreviewContainer.appendChild(wrapper);
        imagePreviewContainer.style.display = "flex";
    };
    reader.readAsDataURL(file);
}

function clearSelectedFile() {
    selectedFile = null;
    imageInput.value = "";
    imagePreviewContainer.innerHTML = "";
    imagePreviewContainer.style.display = "none";
    updateSendBtnState();
}

chatInput.addEventListener("input", updateSendBtnState);

function updateSendBtnState() {
    const hasContent = chatInput.value.trim() || selectedFile;
    if (hasContent) {
        sendBtn.classList.add("ready");
    } else {
        sendBtn.classList.remove("ready");
    }
}

// Send on button click
sendBtn.addEventListener("click", handleSend);

// Send on Enter key
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

async function handleSend() {
    const message = chatInput.value.trim();
    if (!message && !selectedFile) return;

    const currentImage = selectedFile ? URL.createObjectURL(selectedFile) : null;
    const currentFile = selectedFile;
    const currentText = message;

    // Remove welcome message if present
    const welcome = document.querySelector(".welcome");
    if (welcome) welcome.remove();

    // Add user message
    appendMessage("user", { text: currentText, image: currentImage });

    // Preparation for API call
    const formData = new FormData();
    formData.append("message", currentText || "Analyze this leaf image.");
    if (currentFile) {
        formData.append("image", currentFile);
    }

    chatInput.value = "";
    clearSelectedFile();
    setSending(true);

    // Show typing indicator
    const typing = showTypingIndicator();

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) throw new Error("API Error");

        const data = await response.json();
        typing.remove();
        
        appendMessage("bot", {
            text: data.reply,
            gradcam: data.gradcam,
            disease: data.disease,
            confidence: data.confidence,
            originalImage: currentImage
        });
    } catch (err) {
        typing.remove();
        appendMessage("bot", {
            text: "I encountered an error connecting to the AI engine. Please ensure the backend is running.",
            error: true
        });
    } finally {
        setSending(false);
        updateSendBtnState();
    }
}

function appendMessage(role, data) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;

    const content = document.createElement("div");
    content.className = "message-content";

    // User Image
    if (data.image && !data.gradcam) {
        const imgContainer = document.createElement("div");
        const img = document.createElement("img");
        img.className = "chat-image";
        img.src = data.image;
        imgContainer.appendChild(img);
        content.appendChild(imgContainer);
    }

    // GradCAM Panel
    if (data.gradcam && data.originalImage) {
        const panel = document.createElement("div");
        panel.className = "gradcam-panel";
        panel.innerHTML = `
            <div class="panel-header">AI Visual Attention Analysis</div>
            <div class="comparison-grid">
                <div class="comp-item">
                    <span class="comp-label">Input Photography</span>
                    <img src="${data.originalImage}" class="comp-img" />
                </div>
                <div class="comp-item">
                    <span class="comp-label">Attention Heatmap</span>
                    <img src="${data.gradcam}" class="comp-img" />
                </div>
            </div>
            ${data.disease ? `
                <div class="result-card">
                    <div class="result-row">
                        <span class="result-label">Identified Condition:</span>
                        <span class="result-value highlighted">${data.disease}</span>
                    </div>
                </div>
            ` : ""}
        `;
        content.appendChild(panel);
    }

    // Text Content
    if (data.text) {
        const textDiv = document.createElement("div");
        textDiv.className = "message-text";
        // Use marked for markdown rendering
        textDiv.innerHTML = marked.parse(data.text);
        content.appendChild(textDiv);
    }

    wrapper.appendChild(content);
    chatArea.appendChild(wrapper);
    scrollToBottom();
}

function showTypingIndicator() {
    const indicator = document.createElement("div");
    indicator.className = "message bot";
    indicator.innerHTML = `
        <div class="message-content typing">
            <div class="dot"></div><div class="dot"></div><div class="dot"></div>
        </div>
    `;
    chatArea.appendChild(indicator);
    scrollToBottom();
    return indicator;
}

function setSending(isSending) {
    sendBtn.disabled = isSending;
    chatInput.disabled = isSending;
    uploadBtn.disabled = isSending;
    if (!isSending) chatInput.focus();
}

function scrollToBottom() {
    chatArea.scrollTop = chatArea.scrollHeight;
}

