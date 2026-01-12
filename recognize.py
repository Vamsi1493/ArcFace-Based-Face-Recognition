import numpy as np
import joblib
from deepface import DeepFace
from scipy.spatial.distance import cosine

MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"
THRESHOLD = 0.35  # ArcFace cosine threshold (recommended)

embeddings_db = joblib.load("models/arcface_embeddings.pkl")
labels_db = joblib.load("models/arcface_labels.pkl")

def recognize_face(img_path):
    embedding = DeepFace.represent(
        img_path=img_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=True
    )[0]["embedding"]

    distances = [cosine(embedding, db_emb) for db_emb in embeddings_db]
    min_idx = np.argmin(distances)

    if distances[min_idx] < THRESHOLD:
        return labels_db[min_idx], distances[min_idx]
    else:
        return "Unknown", distances[min_idx]

# ---------------- TEST ---------------- #
name, score = recognize_face("dataset\Andy Samberg\Andy Samberg_0.jpg")
print(f"👤 Identity: {name}, Distance: {score:.3f}")
