import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_prep import load_isl_dataset
import matplotlib.pyplot as plt

# --- Load Data ---
DATA_PATH = "../data/ISL_Dataset"
X_train, X_test, y_train, y_test, label_binarizer = load_isl_dataset(DATA_PATH)

# --- Build CNN Model ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_binarizer.classes_), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- Callbacks ---
if not os.path.exists("../model"):
    os.makedirs("../model")

checkpoint = ModelCheckpoint("../model/cnn_isl_model.h5", save_best_only=True, monitor='val_accuracy')
early_stop = EarlyStopping(monitor='val_accuracy', patience=3)

# --- Train ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32,
    callbacks=[checkpoint, early_stop]
)

# --- Plot Results ---
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Training Accuracy")
plt.show()
