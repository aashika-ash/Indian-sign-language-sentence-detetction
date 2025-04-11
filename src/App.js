import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [predictedText, setPredictedText] = useState("");
  const [currentMode, setCurrentMode] = useState(null);
  const [targetMode, setTargetMode] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const letterApiUrl = "http://localhost:5000";
  const sentenceApiUrl = "http://localhost:5001";

  const stopCamera = async (mode) => {
    try {
      const apiUrl = mode === "letter" ? letterApiUrl : sentenceApiUrl;
      const response = await fetch(`${apiUrl}/stop-camera`, { method: "POST" });
      return await response.json();
    } catch (err) {
      console.error("Error stopping camera:", err);
      setError("Failed to stop camera. Please try again.");
      throw err;
    }
  };

  const startMode = async (mode) => {
    setIsLoading(true);
    setTargetMode(mode); // Track the intended mode
    setError(null);
    const apiUrl = mode === "letter" ? letterApiUrl : sentenceApiUrl;

    try {
      if (currentMode) {
        await stopCamera(currentMode);
      }

      const response = await fetch(`${apiUrl}/start-camera`, { method: "POST" });
      await response.json();

      setIsCameraOpen(true);
      setCurrentMode(mode);
      setPredictedText(""); // Clear previous predictions when switching modes
    } catch (err) {
      console.error("Error starting mode:", err);
      setError(`Failed to start ${mode === "letter" ? "Letter" : "Sentence"} mode. Please try again.`);
    } finally {
      setIsLoading(false);
    }
  };

  const clearText = async () => {
    try {
      const apiUrl = currentMode === "letter" ? letterApiUrl : sentenceApiUrl;
      await fetch(`${apiUrl}/clear-text`, { method: "POST" });
      setPredictedText("");
    } catch (err) {
      console.error("Error clearing text:", err);
      setError("Failed to clear text. Please try again.");
    }
  };

  useEffect(() => {
    let interval;

    const fetchPrediction = async () => {
      try {
        const apiUrl = currentMode === "letter" ? letterApiUrl : sentenceApiUrl;
        const response = await fetch(`${apiUrl}/predict`);
        const data = await response.json();
        setPredictedText(data.sentence || "");
        setError(null);
      } catch (err) {
        console.error("Error fetching prediction:", err);
        setError("Failed to get prediction. Please check your connection.");
      }
    };

    if (isCameraOpen && currentMode) {
      fetchPrediction(); // Initial fetch
      interval = setInterval(fetchPrediction, 1000);
    }

    return () => clearInterval(interval);
  }, [isCameraOpen, currentMode]);

  const getVideoFeedUrl = () => {
    if (!currentMode) return "";
    return currentMode === "letter" 
      ? `${letterApiUrl}/video_feed`
      : `${sentenceApiUrl}/video_feed`;
  };

  return (
    <div className="App">
      <div className="App-header">
        <h1 className="heading">Indian Sign Language Detector</h1>

        {error && (
          <div className="error-message">
            <p>{error}</p>
          </div>
        )}

        <div className="button-container">
          <button
            className={`b1 ${currentMode === "letter" ? "active" : ""}`}
            onClick={() => startMode("letter")}
            disabled={isLoading}
          >
            {isLoading && targetMode === "letter" ? "Loading..." : "Letter Level"}
          </button>
          <button
            className={`b2 ${currentMode === "sentence" ? "active" : ""}`}
            onClick={() => startMode("sentence")}
            disabled={isLoading}
          >
            {isLoading && targetMode === "sentence" ? "Loading..." : "Sentence Level"}
          </button>
          <button
            className="b3"
            onClick={clearText}
            disabled={!isCameraOpen || isLoading}
          >
            Clear Text
          </button>
        </div>

        {isCameraOpen && (
          <div className="camera-container">
            <img
              src={getVideoFeedUrl()}
              alt="Camera Feed"
              onError={(e) => {
                console.log("Error loading camera feed:", e);
                setError("Failed to load camera feed. Please check your connection.");
              }}
            />
          </div>
        )}

        {predictedText && (
          <div className="prediction-container">
            <h2>Predicted {currentMode === "letter" ? "Letter" : "Sentence"}:</h2>
            <p>{predictedText}</p>
          </div>
        )}

        {isLoading && targetMode && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <p>Switching to {targetMode === "letter" ? "Letter" : "Sentence"} mode...</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
