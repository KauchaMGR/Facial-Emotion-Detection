
import tensorflow as tf
from pathlib import Path

from load_dataset import train_ds, val_ds, class_names
from model import model

# ==========================================
# 1. Project paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "emotion_model_stage1.keras"

# ==========================================
# 2. Calculate class weights
# ==========================================

class_counts = []

for emotion in class_names:

    emotion_dir = BASE_DIR / "dataset" / "fer2013" / "train" / emotion

    count = len([
        file for file in emotion_dir.iterdir()
        if file.is_file()
    ])

    class_counts.append(count)

total_samples = sum(class_counts)
num_classes = len(class_names)

class_weights = {}

for i, count in enumerate(class_counts):

    weight = total_samples / (num_classes * count)

    class_weights[i] = weight

print("\nClass weights:")

for i, emotion in enumerate(class_names):

    print(f"{emotion}: {class_weights[i]:.4f}")

# ==========================================
# 3. Training configuration
# ==========================================

EPOCHS = 15

# ==========================================
# 4. Callbacks
# ==========================================

callbacks = [

    # Save the best model according to validation loss
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    # Stop if validation loss doesn't improve
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    # Reduce learning rate when validation loss plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]

# ==========================================
# 5. Train the model
# ==========================================

print("\nStarting Stage 1 training...")
print("EfficientNetB0 is frozen.")
print("Only the classification head is trainable.")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==========================================
# 6. Save final training model
# ==========================================

FINAL_MODEL_PATH = MODEL_DIR / "emotion_model_stage1_final.keras"

model.save(FINAL_MODEL_PATH)

print("\nTraining completed!")

print(f"\nBest model saved at:")
print(BEST_MODEL_PATH)

print(f"\nFinal model saved at:")
print(FINAL_MODEL_PATH)
