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
└── README.md                  # You're here!
