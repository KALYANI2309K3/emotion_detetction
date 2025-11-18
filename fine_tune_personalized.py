import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

BASE_PATH = r"C:\Users\kalya\OneDrive\Desktop\Machine Learning\maitri\personalized_dataset"
IMG_SIZE = 48
BATCH_SIZE = 16

EMOTIONS = ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

EMOTION_TO_ID = {}
for i, e in enumerate(EMOTIONS):
    EMOTION_TO_ID[e] = i



def load_personal_dataset(base_path):
    images = []
    labels = []

    for person in os.listdir(base_path):
        person_folder = os.path.join(base_path, person)

        if not os.path.isdir(person_folder):
            continue
        
        print(f"\n📂 Scanning {person} ...")

        for emotion in EMOTIONS:
            emotion_folder = os.path.join(person_folder, emotion)

            if not os.path.isdir(emotion_folder):
                print(f"⚠️ Missing: {person}/{emotion}")
                continue

            for img_name in os.listdir(emotion_folder):
                img_path = os.path.join(emotion_folder, img_name)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print("⚠️ Skipped unreadable:", img_path)
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0

                images.append(img)
                labels.append(EMOTION_TO_ID[emotion])

    return np.array(images), np.array(labels)


print("Loading personalized images...")
X, y = load_personal_dataset(BASE_PATH)

print("\nLoaded:", len(X), "images")

if len(X) == 0:
    raise ValueError("❌ ERROR: No images loaded. Fix folder paths or names.")

X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = to_categorical(y, num_classes=len(EMOTIONS))

print("\nLoading base emotion model...")
model = load_model("best_emotion_model1.h5")

# Freeze early layers
for layer in model.layers[:-3]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(
    "personalized_emotion_model.h5",
    monitor="loss",
    mode="min",
    save_best_only=True
)

early_stop = EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X, y,
    batch_size=BATCH_SIZE,
    epochs=20,
    shuffle=True,
    callbacks=[checkpoint, early_stop]
)

print("\n🎉 Personalized fine-tuning complete!")
print("Saved as: personalized_emotion_model.h5")
