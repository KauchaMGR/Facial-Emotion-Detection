
import cv2
import mediapipe as mp
import numpy as np

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"


class FaceDetector:

    def __init__(self, num_faces=5):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=num_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )

        self.detector = vision.FaceLandmarker.create_from_options(
            options
        )

    def detect_faces(self, image_path):

        # 1. Read image
        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("Could not load input image")

        height, width, _ = image.shape

        print("Image shape:", image.shape)

        # 2. Convert BGR to RGB
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # 3. Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        # 4. Perform landmark detection
        results = self.detector.detect(mp_image)

        print(
            "Detection results:",
            len(results.face_landmarks)
        )

        detected_faces = []

        # 5. Process every detected face
        for i, face_landmarks in enumerate(
            results.face_landmarks
        ):

            print(
                f"Face {i}: {len(face_landmarks)} landmarks detected"
            )

            landmarks = []

            for landmark in face_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                landmarks.append([x, y])

            landmarks = np.array(landmarks)

            detected_faces.append(landmarks)

            # 6. Draw facial landmarks
            for x, y in landmarks:

                if 0 <= x < width and 0 <= y < height:

                    cv2.circle(
                        image,
                        (x, y),
                        2,
                        (0, 255, 0),
                        -1
                    )

            # 7. Calculate bounding box
            x_min = np.min(landmarks[:, 0])
            y_min = np.min(landmarks[:, 1])

            x_max = np.max(landmarks[:, 0])
            y_max = np.max(landmarks[:, 1])

            # 8. Draw bounding box
            cv2.rectangle(
                image,
                (x_min, y_min),
                (x_max, y_max),
                (0, 0, 255),
                2
            )

            # 9. Add detection label
            cv2.putText(
                image,
                "Face Detected",
                (x_min, max(20, y_min - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        return image, detected_faces

    def close(self):

        self.detector.close()


if __name__ == "__main__":

    detector = FaceDetector(num_faces=5)

    try:

        annotated_image, faces = detector.detect_faces(
            "input/test_face.jpg"
        )

        print(
            "Number of detected faces:",
            len(faces)
        )

        cv2.imwrite(
            "output/detected_faces/detected.jpg",
            annotated_image
        )

        print("Landmark detection completed!")

    finally:

        detector.close()
