from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import base64
import os
app = Flask(__name__)

# Configuration matching training
IMG_SIZE = 96

# Prediction cache (hash->(emotion,confidence,box)) to reduce duplicate work
from functools import lru_cache
import hashlib

print("Loading model or interpreter...")
use_tflite = False
model = None
interpreter = None
input_details = output_details = None

# Try to load a TFLite file first for faster inference on CPUs
if os.path.exists('emotion_model.tflite'):
    try:
        interpreter = tf.lite.Interpreter(model_path='emotion_model.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        use_tflite = True
        print("Loaded TFLite interpreter successfully!")
    except Exception as e:
        print(f"Failed to load TFLite model: {e}")
        interpreter = None

if not use_tflite:
    try:
        # Use standard .keras format
        model = tf.keras.models.load_model('emotion_model.keras')
        print("Keras model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Helper for computing prediction hash

def _hash_image(img_bytes: bytes) -> str:
    return hashlib.sha256(img_bytes).hexdigest()

# Fast face detection by working on a reduced size copy

def detect_faces_fast(gray):
    # downscale if too large to speed up cascade
    h, w = gray.shape[:2]
    scale = 1.0
    if w > 640:
        scale = 640.0 / w
        small = cv2.resize(gray, (640, int(h * scale)))
    else:
        small = gray
    faces = face_cascade.detectMultiScale(small, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return []
    # rescale boxes back to original size
    reversed_boxes = []
    for (x, y, fw, fh) in faces:
        reversed_boxes.append((int(x / scale), int(y / scale), int(fw / scale), int(fh / scale)))
    return reversed_boxes

# Prediction wrapper that handles both keras and tflite

def _predict_array(img_arr: np.ndarray):
    """img_arr should be a (1,IMG_SIZE,IMG_SIZE,3) preprocessed float32 array"""
    if use_tflite and interpreter is not None:
        interpreter.set_tensor(input_details[0]['index'], img_arr)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])
    elif model is not None:
        preds = model.predict(img_arr, verbose=0)
    else:
        raise RuntimeError("No model or interpreter available for prediction")
    return preds

# small in-memory cache
prediction_cache = {}

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# The emotion labels (must match the alphabetical order from the training directory `Test`)
emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server.'}), 500

    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided.'}), 400

        # Decode base64 & compute hash for caching
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        key = _hash_image(image_bytes)
        if key in prediction_cache:
            # return cached result
            return jsonify(prediction_cache[key])

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image data.'}), 400

        # convert to grayscale for face detection only
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detect_faces_fast(gray_frame)
        if len(faces) == 0:
            return jsonify({'error': 'No face detected in the image.'})

        # choose largest face
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

        roi_color = frame[y:y+h, x:x+w]
        
        # Convert BGR to RGB since MobileNetV2 and Keras expect RGB images
        roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        
        roi_resized = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
        roi_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.asarray(roi_resized, dtype=np.float32))
        roi_reshaped = np.expand_dims(roi_array, axis=0)

        prediction = _predict_array(roi_reshaped)
        max_index = int(np.argmax(prediction))
        predicted_emotion = emotion_labels[max_index]
        confidence = float(np.max(prediction))

        result = {
            'emotion': predicted_emotion,
            'confidence': round(confidence * 100, 1),
            'face_box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        }

        prediction_cache[key] = result
        # bound cache size to avoid memory blowup
        if len(prediction_cache) > 256:
            prediction_cache.pop(next(iter(prediction_cache)))

        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
