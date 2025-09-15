# ✋ SignBridge – Real-Time Indian Sign Language (ISL) to Text Converter

SignBridge is an AI-powered project designed to bridge the communication gap between the deaf/mute community and the hearing population.  
It uses **Mediapipe Hand Tracking** to detect and extract 3D hand landmarks (x, y, z coordinates), combined with a **deep learning classifier (CNN/MLP)** to recognize Indian Sign Language (ISL) gestures and convert them into **real-time text output**.

---

## 🚀 Features
- 🔹 **Real-time ISL gesture recognition** using webcam.
- 🔹 **Mediapipe-based hand landmark extraction** for accuracy and efficiency.
- 🔹 **Deep learning classifier** trained on landmark data.
- 🔹 **Scalable dataset format** (x, y, z coordinates) for robustness.
- 🔹 **Cross-platform support** (runs on CPU/GPU).
- 🔹 Extensible for **more ISL signs** in future.

---

## 🎯 Motivation
Communication between hearing-impaired individuals and others is often challenging.  
Existing solutions are either expensive or limited in accuracy.  
**SignBridge** aims to provide an **accessible, low-cost, and real-time ISL to text solution** that can run on consumer devices.

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries:** Mediapipe, OpenCV, NumPy, Pandas, TensorFlow/Keras, Scikit-learn  
- **Models:** CNN / MLP classifiers on hand landmark data  
- **Dataset:** Custom-collected ISL signs with (x, y, z) hand coordinates  

---

## 📂 Project Workflow
1. **Data Collection**  
   - Capture ISL gestures using webcam.  
   - Extract hand landmarks `(x, y, z)` via Mediapipe.  
   - Store dataset in `.csv` format.  

2. **Model Training**  
   - Preprocess dataset (normalize coordinates).  
   - Train CNN/MLP classifier for gesture recognition.  
   - Evaluate accuracy, precision, recall.  

3. **Real-Time Inference**  
   - Use webcam to capture hand gestures.  
   - Extract landmarks in real-time.  
   - Predict corresponding ISL gesture → Display as text.  

---

## 📸 Screenshots / Demo
*(Add screenshots or GIFs of your app running here)*  

---

## 📦 Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/SignBridge.git
cd SignBridge

