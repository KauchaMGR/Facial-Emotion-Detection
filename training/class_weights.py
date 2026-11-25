from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Number of images in each class
class_counts = {
    "Angry": 3995,
    "Disgust": 436,
    "Fear": 4097,
    "Happy": 7215,
    "Neutral": 4965,
    "Sad": 4830,
    "Surprise": 3171
}

# Keep the same order as TensorFlow class_names
class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# Create numerical labels
labels = np.arange(len(class_names))

# Calculate weights
weights = compute_class_weight(
    class_weight="balanced",
    classes=labels,
    y=np.concatenate([
        np.full(class_counts[name], i)
        for i, name in enumerate(class_names)
    ])
)

# Convert to dictionary
class_weights = {
    i: float(weights[i])
    for i in range(len(class_names))
}

print("\nClass weights:")

for i, emotion in enumerate(class_names):
    print(f"{emotion:10}: {class_weights[i]:.4f}")