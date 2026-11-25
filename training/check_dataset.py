from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "fer2013"

print("Dataset directory:", DATASET_DIR)
print("-" * 50)

if not DATASET_DIR.exists():
    print("Dataset directory does not exist!")
    exit()

for folder in DATASET_DIR.iterdir():

    if folder.is_dir():

        images = [
            file for file in folder.rglob("*")
            if file.suffix.lower() in
            [".jpg", ".jpeg", ".png", ".bmp"]
        ]

        print(f"{folder.name}: {len(images)} images")

print("\nDataset inspection complete.")