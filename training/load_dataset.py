
import tensorflow as tf
from pathlib import Path

# ----------------------------------
# 1. Project paths
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "fer2013"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

# ----------------------------------
# 2. Configuration
# ----------------------------------

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ----------------------------------
# 3. Load training dataset
# ----------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="int",
    shuffle=True,
    seed=42
)

# ----------------------------------
# 4. Load validation dataset
# ----------------------------------

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="int",
    shuffle=False
)

# ----------------------------------
# 5. Load test dataset
# ----------------------------------

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="int",
    shuffle=False
)

# ----------------------------------
# 6. Class names
# ----------------------------------

class_names = train_ds.class_names

print("\nEmotion classes:")
print(class_names)

# ----------------------------------
# 7. Optimize dataset loading
# ----------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# ----------------------------------
# 8. Inspect dataset
# ----------------------------------

for images, labels in train_ds.take(1):

    print("\nImage batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    print("Image data type:", images.dtype)
    print("Minimum pixel value:", tf.reduce_min(images).numpy())
    print("Maximum pixel value:", tf.reduce_max(images).numpy())

    print("\nFirst 10 labels:")

    for i in range(10):

        label = labels[i].numpy()

        print(
            f"Label: {label}, "
            f"Emotion: {class_names[label]}"
        )

print("\nDataset loading completed successfully!")
