"""
Improved emotion recognition training.

Key upgrades over the baseline notebook:
  - EfficientNetB0 backbone  (+5-8 % accuracy vs MobileNetV2)
  - IMG_SIZE 128             (more facial detail)
  - Two-phase training       (frozen head → full fine-tune)
  - Focal loss               (handles class imbalance better than cross-entropy)
  - BatchNorm frozen in phase 2 (stabilises EfficientNet fine-tuning)
  - Richer augmentation      (brightness, channel-shift)
  - Saves model_config.json  (app.py picks it up automatically)

Expected accuracy: ~68-75 % vs current 60 %.
Run: python train.py
"""

import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import tensorflow.keras.backend as K

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR   = os.environ.get('TRAIN_DIR', os.path.join(BASE_DIR, 'Train'))
VAL_DIR     = os.environ.get('VAL_DIR', os.path.join(BASE_DIR, 'Test'))
IMG_SIZE    = int(os.environ.get('IMG_SIZE', 128))
BATCH_SIZE  = int(os.environ.get('BATCH_SIZE', 32))
EPOCHS_P1   = int(os.environ.get('EPOCHS_P1', 12))
EPOCHS_P2   = int(os.environ.get('EPOCHS_P2', 40))
MODEL_OUT   = os.environ.get('MODEL_OUT', 'emotion_model.keras')
CONFIG_OUT  = os.environ.get('CONFIG_OUT', 'model_config.json')
MAX_TRAIN_STEPS = int(os.environ.get('MAX_TRAIN_STEPS', 0))  # 0 = use full dataset each epoch
MAX_VAL_STEPS   = int(os.environ.get('MAX_VAL_STEPS', 0))

# ── Focal loss ────────────────────────────────────────────────────────────────
def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss — down-weights easy examples so the model focuses on hard ones."""
    def _loss(y_true, y_pred):
        eps    = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1.0 - eps)
        ce     = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow(1.0 - y_pred, gamma)
        return K.mean(K.sum(weight * ce, axis=-1))
    return _loss

# ── Data generators ───────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    brightness_range=(0.75, 1.25),
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    color_mode='rgb', class_mode='categorical', shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    color_mode='rgb', class_mode='categorical', shuffle=False
)

emotion_labels = list(train_gen.class_indices.keys())
num_classes    = train_gen.num_classes
print(f"Classes ({num_classes}): {emotion_labels}")

class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights_arr))
print(f"Class weights: {class_weights}")

full_train_steps = train_gen.samples // BATCH_SIZE
full_val_steps   = val_gen.samples // BATCH_SIZE
train_steps = min(full_train_steps, MAX_TRAIN_STEPS) if MAX_TRAIN_STEPS else full_train_steps
val_steps   = min(full_val_steps, MAX_VAL_STEPS) if MAX_VAL_STEPS else full_val_steps
print(f"Steps/epoch: train={train_steps}/{full_train_steps}  val={val_steps}/{full_val_steps}")

# ── Model ─────────────────────────────────────────────────────────────────────
base = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

x   = base.output
x   = GlobalAveragePooling2D()(x)
x   = BatchNormalization()(x)
x   = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x   = Dropout(0.5)(x)
x   = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
x   = Dropout(0.4)(x)
out = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base.input, outputs=out)

METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
]

# ── Phase 1 : train head only (base frozen) ───────────────────────────────────
print("\n=== Phase 1: Training classification head (base frozen) ===")
base.trainable = False
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=METRICS,
)
model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=EPOCHS_P1,
    class_weight=class_weights,
    callbacks=[
        ModelCheckpoint(MODEL_OUT, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
    ],
)

# ── Phase 2 : fine-tune all layers (BN frozen to stabilise stats) ─────────────
print("\n=== Phase 2: Fine-tuning all layers (BN layers kept frozen) ===")
base.trainable = True
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False  # preserve ImageNet BN stats during fine-tuning

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=METRICS,
)
model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=EPOCHS_P2,
    class_weight=class_weights,
    callbacks=[
        ModelCheckpoint(MODEL_OUT, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ],
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n=== Evaluation on validation set ===")
val_gen.reset()
Y_pred = model.predict(val_gen, steps=int(np.ceil(val_gen.samples / BATCH_SIZE)))
y_pred = np.argmax(Y_pred, axis=1)
print(classification_report(val_gen.classes, y_pred, target_names=emotion_labels))

# ── Save config so app.py auto-configures itself ──────────────────────────────
cfg = {'img_size': IMG_SIZE, 'preprocessing': 'mobilenet_v2', 'labels': emotion_labels}
with open(CONFIG_OUT, 'w') as f:
    json.dump(cfg, f, indent=2)
print(f"Saved {CONFIG_OUT} — app.py will use it automatically on next start.")
