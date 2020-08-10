import cv2
import logging as log
import datetime as dt
from time import sleep
from imageai.Prediction.Custom import CustomImagePrediction

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("model_ex-046_acc-0.967188.h5")
prediction.setJsonPath("model_class.json")
prediction.loadModel(num_objects=3)

def face_detection():
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    log.basicConfig(filename='webcam.log',level=log.INFO)

    video_capture = cv2.VideoCapture(0)
    anterior = 0

    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            pass

        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(120, 120)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            result = frame[y: y + h,
                            x: x + w]

            if len(result) > 0:
                image = result
                image_resize = cv2.resize(image, (224, 224))
                predicted_result = prediction.predictImage(image_resize, result_count=3, input_type="array")
                color = (0, 255, 0)
                predicted_data = str(predicted_result[0][0]) + " " + str(predicted_result[1][0])
                cv2.putText(frame, predicted_data, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

face_detection()