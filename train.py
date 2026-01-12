import os
import numpy as np
import joblib
import tensorflow as tf
from deepface import DeepFace

# ---------------- GPU CHECK & CONFIG ---------------- #
print("🔍 Checking GPU availability...")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print(f"✅ GPU detected: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ No GPU detected. Running on CPU.")

# ---------------- CONFIG ---------------- #
DATASET_PATH = "dataset"
MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"

embeddings_db = []
labels_db = []

print("🚀 Extracting ArcFace embeddings (GPU accelerated if available)...")

# ---------------- EMBEDDING EXTRACTION ---------------- #
for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        try:
            embedding = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR,
                enforce_detection=True,
                align=True,
                normalization="ArcFace"
            )[0]["embedding"]

            embeddings_db.append(embedding)
            labels_db.append(person_name)

        except Exception as e:
            print(f"❌ Skipped {img_path}: {e}")

# ---------------- CONVERT TO NUMPY ---------------- #
embeddings_db = np.array(embeddings_db)
labels_db = np.array(labels_db)

print(f"✅ Total embeddings stored: {embeddings_db.shape[0]}")
print(f"🧠 Embedding dimension: {embeddings_db.shape[1]}")

# ---------------- SAVE EMBEDDINGS ---------------- #
os.makedirs("models", exist_ok=True)

joblib.dump(embeddings_db, "models/arcface_embeddings.pkl")
joblib.dump(labels_db, "models/arcface_labels.pkl")

print("✅ Embedding database saved successfully.")
