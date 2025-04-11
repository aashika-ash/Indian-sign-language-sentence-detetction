import os
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
import time
from collections import deque
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
import string

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize Text-to-Speech (TTS) engine
def speak(predicted_label):
    """Run text-to-speech in a separate thread to prevent blocking."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(predicted_label)
    engine.runAndWait()

# Model path
MODEL_PATH = r"C:\Users\dell\islbridge\backend\models\model.keras"

# Ensure model exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load trained model
model = load_model(MODEL_PATH)

# Define sign language alphabet (1-9 + A-Z)
alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9'] + list(string.ascii_uppercase)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Global variables
sentence = []
last_prediction = None
cap = None
frame_counter = 0
last_speak_time = 0  # Prevents quick repeated speech
prediction_queue = deque(maxlen=3)  # Stores last 3 predictions for smoothing

def init_camera():
    """Initialize the camera if not already open."""
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

def extract_hand_landmarks(image, landmarks):
    """Extract hand landmark coordinates (x, y) from MediaPipe results."""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_list = []

    for landmark in landmarks.landmark:
        x = min(int(landmark.x * image_width), image_width - 1)
        y = min(int(landmark.y * image_height), image_height - 1)
        landmark_list.append([x, y])

    return landmark_list

def normalize_landmarks(landmarks):
    """Normalize landmark coordinates relative to the first point."""
    base_x, base_y = landmarks[0]
    normalized_landmarks = [[x - base_x, y - base_y] for x, y in landmarks]
    flattened = [coord for point in normalized_landmarks for coord in point]
    max_value = max(map(abs, flattened), default=1)
    return [val / max_value for val in flattened]

def generate_frames():
    """Capture video frames, process them, and perform gesture recognition."""
    global sentence, last_prediction, cap, frame_counter, last_speak_time, prediction_queue
    
    init_camera()
    
    with mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to capture image from camera!")
                break

            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            frame_counter += 1
            if frame_counter % 5 != 0:  # Skip frames (process every 5th frame)
                continue

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = extract_hand_landmarks(frame, hand_landmarks)
                    processed_landmarks = normalize_landmarks(landmarks)
                    processed_landmarks = np.array([processed_landmarks])

                    prediction = model.predict(processed_landmarks, verbose=0)

                    if np.max(prediction) >= 0.98:
                        predicted_index = np.argmax(prediction)
                        predicted_label = alphabet[predicted_index]

                        # Apply moving average smoothing
                        prediction_queue.append(predicted_label)
                        if len(set(prediction_queue)) == 1:  # If last 3 predictions are the same
                            if predicted_label != last_prediction:
                                sentence.append(predicted_label)
                                last_prediction = predicted_label

                                # Speak with a delay to avoid fast repetition
                                current_time = time.time()
                                if current_time - last_speak_time > 1.5:  # 1.5 seconds delay
                                    threading.Thread(target=speak, args=(predicted_label,)).start()
                                    last_speak_time = current_time

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())

            if len(sentence) > 7:
                sentence = sentence[-7:]

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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
    global sentence, last_prediction
    sentence.clear()
    last_prediction = None
    return jsonify({'message': 'Text cleared successfully'})

@app.route('/stop-camera', methods=['POST'])
def stop_camera():
    """API route to stop the camera."""
    global cap
    if cap and cap.isOpened():
        cap.release()
    return jsonify({'message': 'Camera stopped successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)