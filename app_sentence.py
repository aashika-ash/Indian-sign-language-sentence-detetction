import cv2
import numpy as np
import os
import mediapipe as mp
import pyttsx3
import threading
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
from KeypointsExtraction import draw_landmarks, image_process, keypoint_extraction

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize text-to-speech engine
def speak(predicted_action):
    """Run text-to-speech in a separate thread to prevent blocking."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(predicted_action)
    engine.runAndWait()

# Model path
MODEL_PATH = r"C:\Users\dell\islbridge\backend\models\my_modellast3.keras"

# Ensure model exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load trained model
model = load_model(MODEL_PATH)

# Load dataset actions
DATASET_PATH = r"C:\Users\dell\islbridge\backend\data"
actions = np.array(os.listdir(DATASET_PATH))

# Global variables
sentence = []
last_prediction = None
cooldown_frames = 0
cooldown_threshold = 30
cap = None

def init_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Error: Camera not detected. Check your webcam permissions!")

@app.route('/start-camera', methods=['POST'])
def start_camera():
    """API route to start the camera."""
    init_camera()
    return jsonify({'message': 'Camera started successfully'})

def generate_frames():
    """Capture video frames, process them, and perform gesture recognition."""
    global sentence, last_prediction, cooldown_frames, cap
    
    init_camera()
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.70, min_tracking_confidence=0.70) as holistic:
        keypoints = []
        while True:
            success, image = cap.read()
            if not success:
                print("Error: Failed to capture image from camera!")
                break

            # Process frame
            results = image_process(image, holistic)
            draw_landmarks(image, results)

            # Check if hands are detected
            if results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints.append(keypoint_extraction(results))

                # Predict every 30 frames
                if len(keypoints) == 30 and cooldown_frames == 0:
                    prediction = model.predict(np.array(keypoints)[np.newaxis, :, :])
                    keypoints = []

                    if np.max(prediction) >= 0.98:
                        predicted_action = actions[np.argmax(prediction)]
                        if predicted_action != last_prediction:
                            sentence.append(predicted_action)
                            last_prediction = predicted_action
                            cooldown_frames = cooldown_threshold
                            threading.Thread(target=speak, args=(predicted_action,)).start()

            # Reduce cooldown
            cooldown_frames = max(0, cooldown_frames - 1)

            # Keep only last 7 words
            if len(sentence) > 7:
                sentence = sentence[-7:]

            # Encode frame
            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """API route to stream the camera feed to the frontend."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['GET'])
def get_prediction():
    """API route to fetch detected sentence."""
    return jsonify({'sentence': ' '.join(sentence)})

@app.route('/clear-text', methods=['POST'])
def clear_text():
    """API route to clear the displayed sentence."""
    global sentence, last_prediction, cooldown_frames
    sentence.clear()
    last_prediction = None
    cooldown_frames = 0
    return jsonify({'message': 'Text cleared successfully'})

@app.route('/stop-camera', methods=['POST'])
def stop_camera():
    """API route to stop the camera."""
    global cap
    if cap and cap.isOpened():
        cap.release()
    return jsonify({'message': 'Camera stopped successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)