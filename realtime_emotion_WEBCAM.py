import cv2
import numpy as np
import tensorflow as tf
import time

model = tf.keras.models.load_model("personalized_emotion_model.h5")

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

IMG_SIZE = (48, 48)

def apply_filters(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    normalized = blurred / 255.0
    return np.expand_dims(normalized, axis=-1)

cap = cv2.VideoCapture(0)

last_pred_time = 0
delay = 5  

print("Webcam running... press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Emotion Detection", frame)

    current_time = time.time()
    if current_time - last_pred_time >= delay:

        img = cv2.resize(frame, IMG_SIZE)
        img = apply_filters(img)
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0]
        emotion = EMOTIONS[np.argmax(pred)]

        print("\n========================")
        print("Predicted Emotion:", emotion)
        print("------------------------")
        print("Raw Probabilities with Labels:")
        for label, value in zip(EMOTIONS, pred):
            print(f"{label:10s} : {value:.4f}")
        print("========================")

        last_pred_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
