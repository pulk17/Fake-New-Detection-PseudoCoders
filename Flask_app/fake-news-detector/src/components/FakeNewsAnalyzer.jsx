import React, { useState } from "react";
import axios from "axios";
import "./FakeNewsAnalyzer.css";

const FakeNewsAnalyzer = () => {
  const [inputText, setInputText] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [explanation, setExplanation] = useState("");
  const [shapPlotPath, setShapPlotPath] = useState("");
  const [lowConfidenceMessage, setLowConfidenceMessage] = useState(""); // For low confidence feedback
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleInputChange = (event) => {
    setInputText(event.target.value);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError("");
    setPrediction(null);
    setConfidence(null);
    setExplanation("");
    setShapPlotPath("");
    setLowConfidenceMessage("");

    try {
      const response = await axios.post("http://localhost:5000/analyze", {
        text: inputText,
      });

      const data = response.data;

      if (data.error) {
        throw new Error(data.error);
      }

      if (data.redirect) {
        // Handle low confidence message
        setLowConfidenceMessage(data.message);
      } else {
        setPrediction(data.label);
        setConfidence(data.confidence);
        setExplanation(data.textual_explanation);
        setShapPlotPath(data.plot_path);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="analyzer-container">
      <h1 className="title">Fake News Detector</h1>
      <textarea
        value={inputText}
        onChange={handleInputChange}
        placeholder="Enter news title or text here"
        className="input-textarea"
      />
      <button onClick={handleSubmit} disabled={loading} className="submit-button">
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {error && <p className="error">Error: {error}</p>}

      {lowConfidenceMessage && (
        <div className="low-confidence-message">
          <p>{lowConfidenceMessage}</p>
        </div>
      )}

      {prediction && (
        <div className="prediction-result">
          <h2>Prediction: {prediction}</h2>
          <p>Confidence: {confidence}</p>
          <p
            className="explanation"
            dangerouslySetInnerHTML={{ __html: explanation }}
          ></p>
          <h3>SHAP Explanation Plot:</h3>
          <img
            src={`http://localhost:5000${shapPlotPath}`} 
            alt="SHAP Explanation"
            className="shap-plot"
          />
        </div>
      )}

      <footer className="footer">
        <p>&#169; 2025 <span className="trademark">PseudoCoders</span></p>
      </footer>
    </div>
  );
};

export default FakeNewsAnalyzer;
