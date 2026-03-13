import React from 'react';
import { UploadCloud, Image as ImageIcon, X } from 'lucide-react';
import './ImageUploader.css';

const ImageUploader = ({ onImageSelect, selectedImage, onClear }) => {
  const [dragActive, setDragActive] = React.useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    // Only accept images
    if (!file.type.startsWith('image/')) return;
    
    // Create preview URL
    const previewUrl = URL.createObjectURL(file);
    onImageSelect(file, previewUrl);
  };

  if (selectedImage) {
    return (
      <div className="preview-container animate-fade-in">
        <div className="preview-wrapper">
          <img src={selectedImage.previewUrl} alt="Selected leaf" className="preview-img" />
          <div className="preview-overlay">
            <button type="button" onClick={onClear} className="clear-btn">
              <X size={20} />
              <span>Remove</span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div 
      className={`uploader-container ${dragActive ? 'drag-active' : ''}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input 
        type="file" 
        id="image-upload" 
        accept="image/*" 
        onChange={handleChange} 
        className="file-input"
      />
      <label htmlFor="image-upload" className="uploader-label">
        <div className="uploader-content">
          <div className="icon-bg">
            <UploadCloud size={32} className="upload-icon" />
          </div>
          <h3 className="uploader-title">Drop your leaf image here</h3>
          <p className="uploader-subtitle">or click to browse from your device</p>
          <div className="uploader-formats">
            <ImageIcon size={14} /> <span>Supports JPG, PNG</span>
          </div>
        </div>
      </label>
    </div>
  );
};

export default ImageUploader;
