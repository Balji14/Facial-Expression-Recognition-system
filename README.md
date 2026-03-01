# 🎭 Emotion Recognition Web App

A real-time facial emotion recognition application built with Flask and TensorFlow. This project uses a MobileNetV2 deep learning model to detect and classify facial expressions from webcam feeds or uploaded images.

---

## ✨ Features

- **Real-time Emotion Detection**: Analyze emotions directly from your webcam in real-time
- **Multiple Input Methods**: Supports both live webcam feed and image upload
- **8 Emotion Classes**: 
  - 😠 Anger
  - 🤨 Contempt
  - 😒 Disgust
  - 😨 Fear
  - 😊 Happy
  - 😐 Neutral
  - 😢 Sad
  - 😲 Surprise

- **Optimized Architecture**: 
  - Face detection with OpenCV Haar Cascade
  - Supports both TensorFlow Keras and TensorFlow Lite models
  - Client-side image compression and resizing
  - Prediction caching for improved performance

- **User-Friendly Interface**: Clean, responsive web UI built with HTML5, CSS3, and JavaScript
- **Lightweight Model**: MobileNetV2-based architecture for efficient inference

---

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.x
- Flask
- OpenCV
- NumPy

---

## 🚀 Installation

### 1. Clone the Repository
```bash
cd Emotionrecognization
```

### 2. Create Virtual Environment (Recommended)
**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Trained Model
Place your trained emotion model in the project root directory:
- **Preferred**: `emotion_model.keras` (TensorFlow Keras format)
- **Alternative**: `emotion_model.tflite` (TensorFlow Lite - faster inference)

---

## 🏃 Running the Application

### Start the Flask Server
```bash
python app.py
```

### Access the Web Interface
Open your browser and navigate to:
```
http://localhost:5000
```

### Quick Start
1. Allow camera access when prompted
2. Click "Capture Frame" to analyze emotions in real-time
3. Or upload an image to test the model
4. View the detected emotion and confidence score

---

## 📁 Project Structure

```
Emotionrecognization/
├── app.py                      # Flask application server
├── fer_inference.py            # Inference utilities
├── emotion_model.keras         # Trained emotion recognition model
├── 1.ipynb                     # Jupyter notebook for model training
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── static/                     # Frontend assets
│   ├── script.js              # Frontend JavaScript logic
│   └── style.css              # Styling
│
├── templates/                  # HTML templates
│   └── index.html             # Main web interface
└── logs/                       # Training logs and TensorBoard events
    └── MobileNetV2_1772220144/
```

---

## 🎓 Model Details

- **Architecture**: MobileNetV2 (Transfer Learning)
- **Input Size**: 96x96 pixels
- **Output**: 8 emotion classes with confidence scores
- **Training Dataset**: FER (Facial Expression Recognition)
- **Framework**: TensorFlow/Keras
- **Optimization**: Supports quantization and TFLite conversion

---

## 🔧 Configuration

Key configuration in `app.py`:
```python
IMG_SIZE = 96  # Input image size for the model
```

The application automatically detects and loads:
- `emotion_model.tflite` (if available) - for faster CPU inference
- `emotion_model.keras` (fallback) - standard Keras model format

---

## ⚡ Performance Optimization

### Model-Level Optimizations
- **TFLite Conversion**: Convert Keras model to TensorFlow Lite for faster CPU inference
  ```python
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  tflite_model = converter.convert()
  ```
- **Post-training Quantization**: INT8 quantization for edge devices
- **Pruning**: Reduce model size during training

### Application-Level Optimizations
- **Prediction Caching**: Implemented LRU cache to avoid recomputing identical frames
- **Image Preprocessing**: Client-side resizing and compression reduce payload
- **Face Detection**: Optimized Haar Cascade with frame size reduction
- **Production Server**: Use Gunicorn, Waitress, or uWSGI instead of Flask's development server

### Training Pipeline Improvements
- Replace `ImageDataGenerator` with `tf.data` pipelines for better performance
- Use mixed precision training: `tf.keras.mixed_precision.set_global_policy('mixed_float16')`
- Implement callbacks: `ReduceLROnPlateau`, `EarlyStopping`
- Cache and prefetch datasets with `AUTOTUNE`

---

## 🐛 Troubleshooting

### Model Not Found Error
**Problem**: `Error loading model: emotion_model not found`

**Solution**: Ensure `emotion_model.keras` or `emotion_model.tflite` exists in the project root directory.

### Webcam Permission Denied
**Problem**: Browser shows "Permission denied" for camera

**Solution**: 
- Grant browser permissions for webcam access when prompted
- Refresh the page if permissions aren't requested
- Check browser privacy settings

### Slow Predictions
**Problem**: Predictions take too long

**Solutions**:
- Use TFLite model for faster inference
- Ensure GPU acceleration is available (check CUDA/cuDNN installation)
- Reduce input image size
- Increase face detection optimization

### Model Loading Issues
**Problem**: TensorFlow or model-related errors

**Solutions**:
- Verify TensorFlow version: `pip show tensorflow`
- Reinstall dependencies: `pip install -r requirements.txt`
- Ensure model file is not corrupted
- Check Python version compatibility (3.7+)

---

## 📊 Training Your Own Model

Use the provided Jupyter notebook (`1.ipynb`) to:
1. Load and preprocess the FER (Facial Expression Recognition) dataset
2. Build and train the MobileNetV2 model
3. Evaluate performance on test data
4. Export the model in Keras format
5. (Optional) Convert to TFLite for edge deployment

### Run Jupyter Notebook
```bash
pip install jupyter
jupyter notebook 1.ipynb
```

### Export Model
```python
# Save as Keras model
model.save('emotion_model.keras')

# Convert to TFLite (optional)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('emotion_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## 🚀 Deployment Tips

### Production Server Setup with Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Waitress (Windows-friendly)
```bash
pip install waitress
waitress-serve --port 5000 app:app
```

### Docker Containerization
For consistent deployment across environments:

**Dockerfile Example:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t emotion-recognition .
docker run -p 5000:5000 emotion-recognition
```

---

## 📝 License

This project is provided as-is for educational and research purposes.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs and issues
- Suggest improvements and optimizations
- Submit pull requests with enhancements
- Improve documentation and examples

---

## 📧 Support

For questions or issues:
1. Check the Troubleshooting section above
2. Review the code comments in `app.py` and `fer_inference.py`
3. Examine the Jupyter notebook `1.ipynb` for training details
4. Create an issue in the repository

---

## 🔗 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [FER Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)

---

**Happy Emotion Recognizing! 😊**
