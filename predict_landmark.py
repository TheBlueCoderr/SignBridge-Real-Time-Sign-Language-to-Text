# predict_landmark.py

import cv2
import numpy as np
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model

# --- Load Model and Label Encoder ---
model = load_model("../model/isl_landmark_model.h5")
label_binarizer = joblib.load("../model/label_binarizer.pkl")

# --- Mediapipe Setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# --- Webcam Stream ---
cap = cv2.VideoCapture(0)
print("📷 Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = []
            for lm in hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]  # Keep x, y, z format

            if len(row) == 63:  # 21 landmarks * 3 (x,y,z)
                input_array = np.array(row).reshape(1, -1)
                pred_probs = model.predict(input_array)
                pred_index = np.argmax(pred_probs)
                pred_label = label_binarizer.classes_[pred_index]
                confidence = np.max(pred_probs) * 100

                # Display Prediction
                cv2.putText(frame, f"{pred_label} ({confidence:.1f}%)", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ISL Landmark Predictor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
