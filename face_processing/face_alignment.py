
import cv2
import numpy as np

from pathlib import Path
from face_detector import FaceDetector


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_IMAGE = BASE_DIR / "input" / "test_face.jpg"

DETECTED_DIR = BASE_DIR / "output" / "detected_faces"

ALIGNED_DIR = BASE_DIR / "output" / "aligned_faces"


# Create output directories if they do not exist
DETECTED_DIR.mkdir(parents=True, exist_ok=True)
ALIGNED_DIR.mkdir(parents=True, exist_ok=True)


def align_face(image, landmarks, output_size=224):

    # Calculate eye centers
    left_eye = np.mean(
        landmarks[[33, 133]],
        axis=0
    )

    right_eye = np.mean(
        landmarks[[362, 263]],
        axis=0
    )

    # Source eye positions
    source_points = np.array(
        [left_eye, right_eye],
        dtype=np.float32
    )

    # Target eye positions
    target_points = np.array(
        [
            [76, 72],
            [148, 72]
        ],
        dtype=np.float32
    )

    # Calculate similarity transformation
    transformation_matrix, _ = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.RANSAC
    )

    if transformation_matrix is None:
        raise ValueError("Face alignment failed")

    # Apply transformation
    aligned_face = cv2.warpAffine(
        image,
        transformation_matrix,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return aligned_face


def main():

    detector = FaceDetector(num_faces=5)

    try:

        print("\nStarting face processing...")

        # Check input image
        if not INPUT_IMAGE.exists():
            raise FileNotFoundError(
                f"Input image not found: {INPUT_IMAGE}"
            )

        # Read original image
        image = cv2.imread(str(INPUT_IMAGE))

        if image is None:
            raise ValueError("Could not load input image")

        print("Input image:", INPUT_IMAGE)
        print("Image shape:", image.shape)

        # Detect faces and extract landmarks
        annotated_image, faces = detector.detect_faces(
            str(INPUT_IMAGE)
        )

        print("Total detected faces:", len(faces))

        # Save detected image
        detected_path = DETECTED_DIR / "detected.jpg"

        success = cv2.imwrite(
            str(detected_path),
            annotated_image
        )

        if not success:
            raise IOError("Could not save detected image")

        print("Detected image saved:", detected_path)

        # Stop if no face detected
        if len(faces) == 0:

            print("No faces detected. Alignment skipped.")

            return

        # Align and save every detected face
        for index, landmarks in enumerate(faces):

            aligned_face = align_face(
                image,
                landmarks,
                output_size=224
            )

            aligned_path = (
                ALIGNED_DIR /
                f"face_{index}_aligned.jpg"
            )

            success = cv2.imwrite(
                str(aligned_path),
                aligned_face
            )

            if not success:
                raise IOError(
                    f"Could not save aligned face: {aligned_path}"
                )

            print(
                f"Face {index} aligned successfully!"
            )

            print(
                "Aligned image saved:",
                aligned_path
            )

        print("\nFace processing completed successfully!")

    finally:

        detector.close()


if __name__ == "__main__":
    main()
