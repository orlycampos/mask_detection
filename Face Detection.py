import cv2
import logging as log
import datetime as dt
from time import sleep
from keras.models import load_model
from pandas import np

model = load_model("/Users/camposo/Desktop/Camera Testing/model_ex-002_acc-0.703598.h5")

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
            minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            result = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if len(result) > 0:
                image = result
                image_resize = cv2.resize(image, (224, 224))
                image_reshape = np.reshape(image_resize, [1, 224, 224, 3])
                prediction = model.predict(image_reshape)
                print(prediction)


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

def crop_image(result, image):
    for obj in result:
        for key, val in obj.items():
            if key == "box":
                bounding_box = val
            elif key == "keypoints":
                keypoints = val
        cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255), 2)

        cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
        cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
        cv2.imshow('image', image)
        image_cropped = image[bounding_box[1]: bounding_box[1] + bounding_box[3],
                        bounding_box[0]: bounding_box[0] + bounding_box[2]]
        cv2.imshow('image', image_cropped)
        return image_cropped

face_detection()