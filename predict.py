import cv2
import numpy as np
from tensorflow.keras.models import load_model
from data_prep import load_isl_dataset
from collections import deque

# --- Load the trained CNN model ---
model = load_model("../model/cnn_isl_model.h5")

# --- Reload the label binarizer to decode predictions ---
_, _, _, _, label_binarizer = load_isl_dataset("../data/ISL_Dataset")
print("Label classes:", label_binarizer.classes_)

# --- Prediction history deque for smoothing ---
predictions_history = deque(maxlen=15)

# --- Start webcam ---
cap = cv2.VideoCapture(0)
print("📷 Webcam started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # Mirror image

    # Define ROI box coordinates
    x1, y1, x2, y2 = 350, 100, 600, 350
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    roi_resized = cv2.resize(roi, (64, 64))
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_reshaped = np.expand_dims(roi_normalized, axis=0)

    # Predict with CNN
    preds = model.predict(roi_reshaped, verbose=0)
    confidence = np.max(preds)
    predicted_class = label_binarizer.classes_[np.argmax(preds)]

    # Store prediction history
    predictions_history.append(predicted_class)
    most_common = max(set(predictions_history), key=predictions_history.count)

    # Show prediction if confident enough
    if confidence > 0.90:
        display_text = f"{most_common} ({confidence*100:.1f}%)"
    else:
        display_text = "Waiting..."

    # Show prediction on main frame
    cv2.putText(frame, display_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    # Show zoomed view of preprocessed ROI
    zoom = 4
    roi_display = cv2.resize((roi_normalized * 255).astype("uint8"), (64 * zoom, 64 * zoom))
    cv2.imshow("Preprocessed ROI (Zoomed)", roi_display)

    # Show main webcam frame
    cv2.imshow("ISL Real-Time Prediction", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Clean up ---
cap.release()
cv2.destroyAllWindows()
