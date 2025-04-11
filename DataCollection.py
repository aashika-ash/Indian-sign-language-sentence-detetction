# DataCollection.py
import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from KeypointsExtraction import *
import keyboard
import time  # Added for delay

actions = np.array([ 'i am fine', 'suffering from fever', 'help me','tell me the truth','let him take time' ])
sequences, frames = 30, 30  # Now collecting **30 frames per sequence**
PATH = os.path.join('data')

# Initialize camera and check
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

with mp.solutions.holistic.Holistic(min_detection_confidence=0.70, min_tracking_confidence=0.70) as holistic:
    for action, sequence in product(actions, range(sequences)):
        action_path = os.path.join(PATH, action, str(sequence))
        os.makedirs(action_path, exist_ok=True)  # Create folder structure if it doesn't exist

        # Wait for spacebar press before starting the sequence
        print(f"Press 'Space' to start recording for '{action}' sequence {sequence}")
        while True:
            ret, image = cap.read()
            if not ret:
                print("Failed to grab frame.")
                continue

            results = image_process(image, holistic)
            draw_landmarks(image, results)
            cv2.putText(image, f'Press "Space" to start recording "{action}"', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('Camera', image)

            if keyboard.is_pressed(' '):  # Start recording when space is pressed
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # Start capturing frames automatically for **30 frames**
        for frame in range(frames):
            ret, image = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            results = image_process(image, holistic)
            draw_landmarks(image, results)

            # Display frame with annotation
            cv2.putText(image, f'Recording {action} - Sequence {sequence}, Frame {frame + 1}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('Camera', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Allow quitting anytime
                cap.release()
                cv2.destroyAllWindows()
                exit()

            # Extract and save keypoints
            keypoints = keypoint_extraction(results)
            frame_path = os.path.join(action_path, str(frame))
            np.save(frame_path, keypoints)

            time.sleep(0.1)  # **Slows down frame collection (100ms per frame)**

cap.release()
cv2.destroyAllWindows()

