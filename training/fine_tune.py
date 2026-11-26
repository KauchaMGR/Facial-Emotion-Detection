
import tensorflow as tf
from pathlib import Path

from load_dataset import train_ds, val_ds, class_names

# ==========================================
# 1. Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "trained_models"

STAGE1_MODEL_PATH = MODEL_DIR / "emotion_model_stage1.keras"

STAGE2_MODEL_PATH = MODEL_DIR / "emotion_model_finetuned.keras"

# ==========================================
# 2. Load best Stage 1 model
# ==========================================

model = tf.keras.models.load_model(STAGE1_MODEL_PATH)

print("\nStage 1 model loaded successfully!")

# ==========================================
# 3. Access EfficientNet base model
# ==========================================

base_model = model.get_layer("efficientnetb0")

# Unfreeze base model
base_model.trainable = True

# Freeze most of the CNN
# Only fine-tune the upper layers
for layer in base_model.layers[:-30]:

    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:

    if isinstance(layer, tf.keras.layers.BatchNormalization):

        layer.trainable = False

# ==========================================
# 4. Inspect trainable layers
# ==========================================

print("\nTrainable EfficientNet layers:")

for layer in base_model.layers:

    if layer.trainable:

        print(layer.name)

print("\nTrainable parameters:",
      sum(
          tf.keras.backend.count_params(w)
          for w in model.trainable_weights
      )
)

# ==========================================
# 5. Recompile model
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 6. Class weights
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

class_weights = {
    i: total_samples / (num_classes * count)
    for i, count in enumerate(class_counts)
}

# ==========================================
# 7. Callbacks
# ==========================================

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(STAGE2_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# ==========================================
# 8. Fine-tune
# ==========================================

EPOCHS = 10

print("\nStarting Stage 2 fine-tuning...")
print("EfficientNet upper layers are trainable.")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ==========================================
# 9. Save final fine-tuned model
# ==========================================

FINAL_PATH = MODEL_DIR / "emotion_model_finetuned_final.keras"

model.save(FINAL_PATH)

print("\nFine-tuning completed!")

print("Best model saved at:")
print(STAGE2_MODEL_PATH)

print("Final model saved at:")
print(FINAL_PATH)