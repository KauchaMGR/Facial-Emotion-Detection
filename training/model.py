
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. Configuration
# ==========================================

IMAGE_SIZE = 224
NUM_CLASSES = 7

EMOTION_CLASSES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# ==========================================
# 2. Load pretrained EfficientNetB0
# ==========================================

base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

# Freeze the pretrained CNN initially
base_model.trainable = False

# ==========================================
# 3. Build emotion classification model
# ==========================================

inputs = layers.Input(
    shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    name="input_image"
)

# Extract features using EfficientNet
x = base_model(inputs, training=False)

# Convert feature maps into a feature vector
x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

# Regularization
x = layers.Dropout(
    0.4,
    name="dropout"
)(x)

# Fully connected classification layer
x = layers.Dense(
    256,
    activation="relu",
    name="dense_features"
)(x)

# Additional regularization
x = layers.Dropout(
    0.3,
    name="dropout_2"
)(x)

# Output layer: seven emotion probabilities
outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="emotion_output"
)(x)

# ==========================================
# 4. Create final model
# ==========================================

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="EfficientNetB0_Emotion_Classifier"
)

# ==========================================
# 5. Compile model
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 6. Display model information
# ==========================================

model.summary()

print("\nEmotion classes:")
print(EMOTION_CLASSES)

print("\nModel successfully created and compiled!")
