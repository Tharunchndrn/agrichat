import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, Leaf, Image as ImageIcon, X, Loader2 } from 'lucide-react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'bot',
      text: 'Hello! I am AgriChat. Upload a picture of a leaf for a professional diagnosis or ask me any agricultural question.',
    }
  ]);
  const [inputVal, setInputVal] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      e.target.value = null; 
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
  };

  const handleSend = async () => {
    if (!inputVal.trim() && !selectedFile) return;

    const currentImage = previewUrl;
    const currentFile = selectedFile;
    const currentText = inputVal;

    const userMsg = {
      id: Date.now(),
      sender: 'user',
      text: currentText,
      image: currentImage
    };

    setMessages(prev => [...prev, userMsg]);
    setInputVal('');
    clearFile();
    setIsLoading(true);

    try {
      const formData = new FormData();
      if (currentFile) formData.append('image', currentFile);
      formData.append('message', currentText || "Analyze this leaf image.");

      const response = await fetch('http://127.0.0.1:8000/chat', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('API Error');

      const data = await response.json();
      
      const botMsg = {
        id: Date.now() + 1,
        sender: 'bot',
        text: data.reply,
        gradcam: data.gradcam,
        disease: data.disease,
        confidence: data.confidence,
        originalImage: currentImage
      };

      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        sender: 'bot',
        text: "I encountered an error connecting to the AI engine. Please ensure the backend is running.",
        error: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
         <div className="header-brand">
           <div className="logo-badge"><Leaf size={22} /></div>
           <div>
             <h1>AgriChat AI</h1>
             <p className="online-status">Expert System • Online</p>
           </div>
         </div>
      </header>

      <div className="messages-area">
        {messages.map((msg) => (
          <div key={msg.id} className={`message-wrapper ${msg.sender === 'user' ? 'user-wrapper' : 'bot-wrapper'}`}>
            <div className={`message-bubble ${msg.sender === 'user' ? 'user-bubble' : 'bot-bubble'} ${msg.error ? 'error-bubble' : ''}`}>
              
              {/* Image only for user messages */}
              {msg.image && !msg.gradcam && (
                <div className="message-image-container">
                  <img src={msg.image} alt="Crop" className="message-image" />
                </div>
              )}

              {/* Side-by-Side Comparison — ON TOP */}
              {msg.gradcam && msg.originalImage && (
                <div className="gradcam-panel">
                  <div className="panel-header">
                    AI Visual Attention Analysis
                  </div>
                  <div className="comparison-grid">
                    <div className="comp-item">
                      <span className="comp-label">Input Photography</span>
                      <img src={msg.originalImage} alt="Input" className="comp-img" />
                    </div>
                    <div className="comp-item">
                      <span className="comp-label">Attention Heatmap</span>
                      <img src={msg.gradcam} alt="Heatmap" className="comp-img" />
                    </div>
                  </div>
                  
                  {msg.disease && (
                    <div className="result-card">
                      <div className="result-row">
                        <span className="result-label">Identified Condition:</span>
                        <span className="result-value highlighted">{msg.disease}</span>
                      </div>
                      <div className="result-row">
                        <span className="result-label">System Confidence:</span>
                        <span className="result-value">{(msg.confidence * 100).toFixed(2)}%</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              {/* Text response — BELOW the comparison */}
              {msg.text && (
                <div className="message-text">
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="message-wrapper bot-wrapper">
             <div className="message-bubble bot-bubble typing">
                <div className="dot"></div><div className="dot"></div><div className="dot"></div>
             </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        {previewUrl && (
          <div className="attach-preview">
             <img src={previewUrl} alt="Preview" />
             <button className="del-attach" onClick={clearFile}><X size={12}/></button>
          </div>
        )}
        
        <div className="bar-controls">
           <label className="icon-btn">
             <input type="file" accept="image/*" onChange={handleFileSelect} hidden />
             <ImageIcon size={20} />
           </label>
           
           <input 
             type="text" 
             className="main-input" 
             placeholder="Discuss symptoms or attach a photo..." 
             value={inputVal}
             onChange={(e) => setInputVal(e.target.value)}
             onKeyPress={(e) => e.key === 'Enter' && handleSend()}
           />
           
           <button 
             className={`act-send ${(inputVal.trim() || previewUrl) && !isLoading ? 'ready' : ''}`}
             onClick={handleSend}
             disabled={(!inputVal.trim() && !previewUrl) || isLoading}
           >
             {isLoading ? <Loader2 className="spinning" size={18} /> : <Send size={18} />}
           </button>
        </div>
      </div>
    </div>
  );
}

export default App;
