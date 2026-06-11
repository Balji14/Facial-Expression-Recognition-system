from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import base64
import os
import json
import hashlib

app = Flask(__name__)

# ── Load model config written by train.py ─────────────────────────────────────
_cfg = {}
if os.path.exists('model_config.json'):
    with open('model_config.json') as f:
        _cfg = json.load(f)

_preprocessing  = _cfg.get('preprocessing', 'mobilenet_v2')
emotion_labels  = _cfg.get('labels', ['anger', 'contempt', 'disgust', 'fear',
                                       'happy', 'neutral', 'sad', 'surprise'])

if _preprocessing == 'efficientnet':
    _preprocess_fn = tf.keras.applications.efficientnet.preprocess_input
else:
    _preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input

# IMG_SIZE is overridden below once the model is loaded (reads actual input shape)
IMG_SIZE = _cfg.get('img_size', 96)

# ── Load model / TFLite interpreter ───────────────────────────────────────────
print("Loading model or interpreter...")
use_tflite = False
model      = None
interpreter = None
input_details = output_details = None

if os.path.exists('emotion_model.tflite'):
    try:
        interpreter = tf.lite.Interpreter(model_path='emotion_model.tflite')
        interpreter.allocate_tensors()
        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        use_tflite = True
        IMG_SIZE   = input_details[0]['shape'][1]
        print(f"Loaded TFLite interpreter. IMG_SIZE={IMG_SIZE}")
    except Exception as e:
        print(f"Failed to load TFLite model: {e}")
        interpreter = None

if not use_tflite:
    try:
        model    = tf.keras.models.load_model('emotion_model.keras', compile=False)
        IMG_SIZE = model.input_shape[1]   # read from model — works for any backbone
        print(f"Keras model loaded. IMG_SIZE={IMG_SIZE}, preprocessing={_preprocessing}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# ── Face detection ────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
prediction_cache: dict = {}


def _hash_image(img_bytes: bytes) -> str:
    return hashlib.sha256(img_bytes).hexdigest()


def detect_faces_fast(gray):
    h, w  = gray.shape[:2]
    scale = 1.0
    if w > 640:
        scale = 640.0 / w
        small = cv2.resize(gray, (640, int(h * scale)))
    else:
        small = gray
    faces = face_cascade.detectMultiScale(
        small, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return []
    return [
        (int(x / scale), int(y / scale), int(fw / scale), int(fh / scale))
        for (x, y, fw, fh) in faces
    ]


def _predict_array(img_arr: np.ndarray) -> np.ndarray:
    if use_tflite and interpreter is not None:
        interpreter.set_tensor(input_details[0]['index'], img_arr)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])
    if model is not None:
        return model.predict(img_arr, verbose=0)
    raise RuntimeError("No model or interpreter available")


def _predict_with_tta(roi_rgb: np.ndarray) -> np.ndarray:
    """2-way TTA: original + horizontal flip, averaged.
    Horizontal flip is safe for facial expressions (symmetric emotions).
    Typically adds +1-2 % accuracy at no retraining cost.
    """
    def _prep(img):
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        arr     = _preprocess_fn(np.asarray(resized, dtype=np.float32))
        return np.expand_dims(arr, axis=0)

    pred_orig    = _predict_array(_prep(roi_rgb))
    pred_flipped = _predict_array(_prep(np.fliplr(roi_rgb)))
    return (pred_orig + pred_flipped) / 2.0


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None and interpreter is None:
        return jsonify({'error': 'Model not loaded on server.'}), 500

    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided.'}), 400

        image_data  = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        key         = _hash_image(image_bytes)
        if key in prediction_cache:
            return jsonify(prediction_cache[key])

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid image data.'}), 400

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces      = detect_faces_fast(gray_frame)
        if len(faces) == 0:
            return jsonify({'error': 'No face detected in the image.'})

        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        roi_rgb    = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)

        prediction = _predict_with_tta(roi_rgb)
        max_index  = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        result = {
            'emotion':    emotion_labels[max_index],
            'confidence': round(confidence * 100, 1),
            'face_box':   {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
        }

        prediction_cache[key] = result
        if len(prediction_cache) > 256:
            prediction_cache.pop(next(iter(prediction_cache)))

        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)
