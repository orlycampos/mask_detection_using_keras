from keras.models import load_model
import cv2
import numpy as np

def camera_mask_detection():
    model = load_model("mask_detection_model_3CATG.h5")

    face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    source = cv2.VideoCapture("video1.MOV")

    labels_dict = {0: 'MASK', 1: 'NO MASK', 2: "INCORRECT"}
    color_dict = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 125, 255)}
    while (True):

        ret, img = source.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_clsfr.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            face_img = gray[y:y + w, x:x + w]

            if len(face_img) > 0:
                resized = cv2.resize(face_img, (128, 128))
                normalized = resized / 255.0
                reshaped = np.reshape(normalized, (1, 128, 128, 1))
                result = model.predict(reshaped)

                label = np.argmax(result, axis=1)[0]

                cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
                cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
                cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)

        if (key == 27):
            break

    cv2.destroyAllWindows()
    source.release()

camera_mask_detection()