import React, { useRef, useCallback } from "react";
import Webcam from "react-webcam";

const WebcamComponent = () => {
  const webcamRef = useRef(null);

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    console.log("Captured Image:", imageSrc);
  }, [webcamRef]);

  return (
    <div>
      <Webcam ref={webcamRef} screenshotFormat="image/jpeg" />
      <button onClick={capture}>Capture & Predict</button>
    </div>
  );
};

export default WebcamComponent;
