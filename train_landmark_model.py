# train_landmark_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load the dataset
df = pd.read_csv("../data/landmark_dataset.csv")

# Features and labels
X = df.drop('label', axis=1).values
y = df['label'].values

# Encode labels
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(26, activation='softmax')  # 26 output classes (A-Z)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Save the model
model.save("../model/isl_landmark_model.h5")

# Save label binarizer for use in prediction
import joblib
joblib.dump(lb, "../model/label_binarizer.pkl")

print("✅ Model and label binarizer saved!")
