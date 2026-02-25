const API_URL = "http://localhost:8000/chat";

const chatArea = document.getElementById("chatArea");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const errorToast = document.getElementById("errorToast");
const imageInput = document.getElementById("imageInput");
const uploadBtn = document.getElementById("uploadBtn");
const imagePreviewContainer = document.getElementById("imagePreviewContainer");

let selectedFile = null;

// File Upload Logic
uploadBtn.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
        if (!file.type.startsWith("image/")) {
            showError("Please select an image file");
            return;
        }
        selectedFile = file;
        showImagePreview(file);
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

    // Remove welcome message if present
    const welcome = document.querySelector(".welcome");
    if (welcome) welcome.remove();

    // Add user message
    appendMessage("user", message, selectedFile);

    // Preparation for API call
    const formData = new FormData();
    formData.append("message", message || "Describe this image for agriculture context.");
    if (selectedFile) {
        formData.append("image", selectedFile);
    }

    chatInput.value = "";
    clearSelectedFile();
    chatInput.focus();
    setSending(true);

    // Show typing indicator
    const typing = showTypingIndicator();

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData, // Send as FormData
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Server error (${response.status})`);
        }

        const data = await response.json();
        typing.remove();
        appendMessage("bot", data.reply);
    } catch (err) {
        typing.remove();
        showError(err.message || "Failed to connect to server");
        appendMessage("bot", "⚠️ Sorry, something went wrong. Please try again.");
    } finally {
        setSending(false);
    }
}

function appendMessage(role, text, file = null) {
    const msg = document.createElement("div");
    msg.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "🧑" : "🌱";

    const content = document.createElement("div");
    content.className = "message-content";

    // If there's an image, render it
    if (file) {
        const img = document.createElement("img");
        img.className = "chat-image";
        img.src = URL.createObjectURL(file);
        content.appendChild(img);
    }

    if (text) {
        const textNode = document.createElement("div");
        textNode.textContent = text;
        content.appendChild(textNode);
    }

    msg.appendChild(avatar);
    msg.appendChild(content);
    chatArea.appendChild(msg);
    scrollToBottom();
}

function showTypingIndicator() {
    const typing = document.createElement("div");
    typing.className = "typing-indicator";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "🌱";

    const dots = document.createElement("div");
    dots.className = "typing-dots";
    dots.innerHTML = "<span></span><span></span><span></span>";

    typing.appendChild(avatar);
    typing.appendChild(dots);
    chatArea.appendChild(typing);
    scrollToBottom();

    return typing;
}

function setSending(isSending) {
    sendBtn.disabled = isSending;
    chatInput.disabled = isSending;
    uploadBtn.disabled = isSending;
    if (!isSending) chatInput.focus();
}

function showError(message) {
    errorToast.textContent = message;
    errorToast.classList.add("show");
    setTimeout(() => errorToast.classList.remove("show"), 4000);
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatArea.scrollTop = chatArea.scrollHeight;
    });
}
