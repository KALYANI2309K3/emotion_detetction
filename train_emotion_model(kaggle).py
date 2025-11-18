import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

KAGGLE_TRAIN = r"C:\Users\kalya\OneDrive\Desktop\Machine Learning\maitri\FER\train"
KAGGLE_VAL   = r"C:\Users\kalya\OneDrive\Desktop\Machine Learning\maitri\FER\validation"
PERSONAL_DATASET = r"C:\Users\kalya\OneDrive\Desktop\Machine Learning\maitri\personalized_dataset"

COMBINED_TRAIN = r"C:\Users\kalya\OneDrive\Desktop\Machine Learning\maitri\combined_train"

if os.path.exists(COMBINED_TRAIN):
    shutil.rmtree(COMBINED_TRAIN)

shutil.copytree(KAGGLE_TRAIN, COMBINED_TRAIN)

# Copy personalized images into combined folder
for emotion in os.listdir(PERSONAL_DATASET):
    src_folder = os.path.join(PERSONAL_DATASET, emotion)
    dst_folder = os.path.join(COMBINED_TRAIN, emotion)

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for img in os.listdir(src_folder):
        shutil.copy(os.path.join(src_folder, img), dst_folder)

print("Combined dataset ready at:", COMBINED_TRAIN)

IMG_SIZE = 48
BATCH_SIZE = 32

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
).flow_from_directory(
    COMBINED_TRAIN,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    KAGGLE_VAL,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

num_classes = train_gen.num_classes

model = Sequential([
    Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint("best_emotion_model.h15", monitor="val_accuracy", save_best_only=True, mode="max")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model.fit(
    train_gen,
    epochs=35,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stop]
)

model.save("final_emotion_model.h5")

print("\nMODEL TRAINING COMPLETE!")
print("Saved as best_emotion_model1.h5 and final_emotion_model.h5")
