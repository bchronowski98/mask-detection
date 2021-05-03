import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

# load models
facemodel = r"files\deploy.prototxt"
weights = r"files\res10_300x300_ssd_iter_140000.caffemodel"

CaffeModel = cv2.dnn.readNet(facemodel, weights)
MaskModel = load_model("maskmodel.model")

capture = cv2.VideoCapture(0)
dim = (400, 400)

while True:

    isTrue, frame = capture.read()
    frame = cv2.flip(frame, 1, 1)

    (image_height, image_width) = frame.shape[:2]

    # set blob and forward pass
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    CaffeModel.setInput(blob)
    detections = CaffeModel.forward()

    faces = []
    coordinates = []
    predicted_faces = []

    for i in range(0, detections.shape[2]):

        # probability
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            # x, y for rectangle

            x1 = int(detections[0, 0, i, 3] * np.array([image_width]))
            y1 = int(detections[0, 0, i, 4] * np.array([image_height]))
            x2 = int(detections[0, 0, i, 5] * np.array([image_width]))
            y2 = int(detections[0, 0, i, 6] * np.array([image_height]))

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_width - 1, x2)
            y2 = min(image_height - 1, y2)

            # preprocess face and add to the list

            face = frame[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = tf.keras.preprocessing.image.img_to_array(face)
            face = tf.keras.applications.xception.preprocess_input(face)

            faces.append(face)
            coordinates.append((x1, y1, x2, y2))

    if len(faces) > 0:

        predicted_faces = np.array(faces, dtype="float32")
        predictions = MaskModel.predict(predicted_faces, batch_size=32)

        for (rectangle, prediction) in zip(coordinates, predictions):
            # unzip corresponding coordinates and prediction
            (x1, y1, x2, y2) = rectangle
            (no_mask, mask) = prediction

            if mask > no_mask:
                label = 'mask'
                color = (0, 255, 0)
            else:
                label = 'no_mask'
                color = (0, 0, 255)

            percent = max(mask, no_mask) * 100
            label = f"{label}--{percent:.2f}"

            # rectangle and text
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_ITALIC, 0.45, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    elif len(faces) <= 0:
        cv2.putText(frame, "no face detected", (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Live", frame)

    # q to exit
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
