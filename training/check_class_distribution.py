
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "fer2013"

splits = ["train", "val", "test"]

emotions = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

for split in splits:

    split_dir = DATASET_DIR / split

    print(f"\n{'=' * 40}")
    print(f"{split.upper()} DATASET")
    print(f"{'=' * 40}")

    total = 0

    for emotion in emotions:

        emotion_dir = split_dir / emotion

        if emotion_dir.exists():

            count = sum(
                1 for file in emotion_dir.iterdir()
                if file.is_file()
            )

            total += count

            print(f"{emotion:10} : {count}")

    print("-" * 40)
    print(f"Total images: {total}")
