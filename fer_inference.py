import cv2
import numpy as np
import tensorflow as tf
import os
# Configuration matching training
IMG_SIZE = 96

# Load the trained model or interpreter (same logic as app.py)
print("Loading model/interpreter...")
use_tflite = False
interpreter = None
input_details = output_details = None
model = None

if os.path.exists('emotion_model.tflite'):
    try:
        interpreter = tf.lite.Interpreter(model_path='emotion_model.tflite')
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        use_tflite = True
        print("TFLite interpreter loaded successfully!")
    except Exception as e:
        print("Failed to load TFLite: ", e)

if not use_tflite:
    try:
        model = tf.keras.models.load_model('emotion_model.keras')
        print("Keras model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you run fer_training.py first to train and save the model!")
        exit()

# helper prediction

def predict_raw(img_arr):
    if use_tflite and interpreter:
        interpreter.set_tensor(input_details[0]['index'], img_arr)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])
    else:
        return model.predict(img_arr, verbose=0)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# The emotion labels (must match the alphabetical order from the training directory `Test`)
emotion_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")

# Create the window and set it to correctly scale in Fullscreen mode
cv2.namedWindow('Facial Expression Recognition', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Facial Expression Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mirror the frame horizontally so it acts like a mirror
    frame = cv2.flip(frame, 1)
    
    # Convert image to grayscale for face detection only
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Crop face using the original COLOR frame because MobileNetV2 needs RGB
        roi_color = frame[y:y+h, x:x+w]
        
        try:
            # Convert BGR (cv2 default) to RGB (Keras default)
            roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            
            # Resize for MobileNetV2
            roi_resized = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
            
            # MobileNetV2 Preprocessing
            roi_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(roi_resized, dtype=np.float32))
            roi_reshaped = np.expand_dims(roi_array, axis=0)
            
            # Make prediction
            prediction = predict_raw(roi_reshaped)
            max_index = int(np.argmax(prediction))
            predicted_emotion = emotion_labels[max_index]
            confidence = np.max(prediction)
            
            # Display label above rectangle
            label = f"{predicted_emotion} ({confidence*100:.1f}%)"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            # Catch resize errors if face goes outside frame bounds
            pass
        
    # Show video frame
    cv2.imshow('Facial Expression Recognition', frame)
    
    # Hit 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
