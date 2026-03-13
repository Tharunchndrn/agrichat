import React from 'react';
import ReactMarkdown from 'react-markdown';
import './ResultsView.css';

const ResultsView = ({ originalImage, result }) => {
  if (!result) return null;

  return (
    <div className="results-container animate-fade-in">
      <div className="results-header">
        <h2 className="results-title">Analysis Complete</h2>
        <div className="status-badge success">Diagnosis Ready</div>
      </div>
      
      <div className="visuals-grid">
        <div className="visual-card">
          <div className="visual-header">
            <h3>Original Image</h3>
          </div>
          <div className="visual-content">
            <img src={originalImage} alt="Original uploaded leaf" />
          </div>
        </div>

        {result.gradcam && (
          <div className="visual-card">
            <div className="visual-header">
              <h3>AI Attention Map (Grad-CAM)</h3>
              <span className="tooltip-icon" title="Shows exactly where the AI looked to make its decision">?</span>
            </div>
            <div className="visual-content">
              <img src={result.gradcam} alt="Grad-CAM visualization" />
            </div>
          </div>
        )}
      </div>

      <div className="expert-analysis glass-panel">
        <div className="analysis-header">
           <img src="https://img.icons8.com/fluency/48/000000/bot.png" alt="AI Expert" className="bot-icon"/>
           <h3>Expert Breakdown</h3>
        </div>
        <div className="markdown-content">
          <ReactMarkdown>{result.reply}</ReactMarkdown>
        </div>
      </div>
    </div>
  );
};

export default ResultsView;
