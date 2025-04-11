Indian Sign Language Sentence Detection

This project aims to detect Indian Sign Language (ISL) gestures for letters, numbers, and sentence-level communication using deep learning models. It combines a React.js frontend with a Flask-based Python backend to provide real-time gesture recognition support.



    
Indian-sign-language-sentence-detection/
│
├── src/                        # Frontend (React.js)
│
├── main.py                    # Backend entry point (Flask)
├── app_sentence.py            # Sentence-level ISL detection API
├── app_letter.py              # Letter/Number ISL detection API
│
├── models/                    # Trained models for:
│   ├── my_modellast3.keras    # Sentence detection (30-frame input)
│   └── model.keras            # Letter/Number detection (single-frame input)
│
├── DataCollection.py          # Script to collect custom ISL gesture dataset
├── modeltraining.py           # Training script for sentence detection model
├── realtime.py                # Run trained model on webcam for real-time predictions
│
├── newmod.ipynb               # Model training notebook using Kaggle ISL datasets
│
└── README.md                  

💡 Features
Real-time prediction of letters, numbers, and sentence-level gestures.

Custom dataset collection and model training pipeline using MediaPipe.

Integrated React frontend and Flask backend.

Trained models saved and reused for fast deployment.

🔧 Backend Components
main.py – Runs the Flask server.

app_sentence.py – Predicts full ISL sentences from 30-frame keypoint sequences.

app_letter.py – Predicts static gestures like letters and numbers from single frames.

📦 Models
The models/ directory contains:

my_modellast3.keras – Trained using dynamic sentence-level gesture videos.

model.keras – Trained on single-frame gestures (letters and numbers).

🛠️ Model Training Scripts
DataCollection.py: Collect your own ISL gesture data using webcam.

modeltraining.py: Train the sentence-level gesture recognition model.

realtime.py: Test the trained model in real-time with webcam.

newmod.ipynb: Jupyter notebook for training on Kaggle datasets (letters and sentences).



✅ CNN + LSTM for sentence-level gesture detection

✅ FCNN (Fully Connected Neural Network) for letter and number recognition
