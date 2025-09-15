# 01_collect_landmarks.py

import cv2
import mediapipe as mp
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.75)
mp_draw = mp.solutions.drawing_utils

# Save data here
SAVE_PATH = "../data/landmark_dataset.csv"
os.makedirs("../data", exist_ok=True)

# Ask for label
label = input("Enter the label (A-Z): ").strip().upper()
print(f"⚠️ Press 'q' to stop recording for letter '{label}'")

data = []
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if results.multi_handedness[i].classification[0].label != "Right":
                continue

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            row = []
            for lm in hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]
            row.append(label)
            data.append(row)

            

    cv2.putText(frame, f"Label: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("Collecting Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
columns = [f"{axis}{i}" for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
df = pd.DataFrame(data, columns=columns)

if os.path.exists(SAVE_PATH):
    df.to_csv(SAVE_PATH, mode='a', header=False, index=False)
else:
    df.to_csv(SAVE_PATH, index=False)

print(f"✅ Saved {len(data)} samples to {SAVE_PATH}")
