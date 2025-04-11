import numpy as np
import os
import cv2
import mediapipe as mp
import pyttsx3
import keyboard
from tensorflow.keras.models import load_model
from KeypointsExtraction import draw_landmarks, image_process, keypoint_extraction

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speaking rate

# Path to data and actions defined during training
PATH = r"C:\Users\dell\project-1\data"
actions = np.array(os.listdir(PATH))

# Load the trained model
model_path = r"C:\Users\dell\islbridge\backend\models\my_modellast3.keras"
model = load_model(model_path)


# Initialize prediction and sentence-related lists
sentence, keypoints, last_prediction = [], [], None
cooldown_frames, cooldown_threshold = 0, 30  # Cooldown period of 30 frames
skip_frames_after_hand_detected, skip_counter = 5, 0  # Skip 5 frames after hand is detected

# Open camera for capturing
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

with mp.solutions.holistic.Holistic(min_detection_confidence=0.70, min_tracking_confidence=0.70) as holistic:
    hand_present = False  # Track if a hand is present

    while cap.isOpened():
        # Capture frame from camera
        ret, image = cap.read()
        if not ret:
            break

        # Process frame and extract keypoints
        results = image_process(image, holistic)
        draw_landmarks(image, results)

        # Check if a hand is present in the frame
        hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

        if hand_detected:
            if not hand_present:
                # Hand just appeared, start skip counter
                hand_present = True
                skip_counter = skip_frames_after_hand_detected
            elif skip_counter > 0:
                # Continue skipping frames for stabilization
                skip_counter -= 1
                continue

            # Extract keypoints after hand has been stable for 5 frames
            keypoints.append(keypoint_extraction(results))

            # Predict every 30 frames if cooldown is not active
            if len(keypoints) == 30 and cooldown_frames == 0:
                keypoints = np.array(keypoints)
                prediction = model.predict(keypoints[np.newaxis, :, :])
                keypoints = []

                # Check if the prediction exceeds threshold
                if np.max(prediction) >= 0.98:
                    predicted_action = actions[np.argmax(prediction)]

                    # Only append if prediction differs from the last one
                    if predicted_action != last_prediction:
                        sentence.append(predicted_action)
                        last_prediction = predicted_action
                        cooldown_frames = cooldown_threshold  # Activate cooldown

                        # Speak the predicted word
                        engine.say(predicted_action)
                        engine.runAndWait()

        else:
            hand_present = False  # Reset if no hand is detected
            keypoints = []  # Clear keypoints when hands go out of frame

        # Decrease cooldown frame count
        cooldown_frames = max(0, cooldown_frames - 1)

        # Limit sentence length for display
        if len(sentence) > 7:
            sentence = sentence[-7:]

        # Reset on Spacebar press
        if keyboard.is_pressed(' '):
            sentence, keypoints, last_prediction = [], [], None

        # Adjust sentence for display
        if sentence:
            sentence[0] = sentence[0].capitalize()

        # Display prediction on image
        display_text = ' '.join(sentence)
        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        cv2.putText(image, display_text, (text_x, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the output
        cv2.imshow('Real-time Sign Prediction', image)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
