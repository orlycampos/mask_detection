import numpy as np
from keras.models import load_model
import cv2

model = load_model("/Users/camposo/Desktop/Camera Testing/model_ex-002_acc-0.703598.h5")

def read_image():
    mask_image = cv2.imread('mask.png')
    image_resize = cv2.resize(mask_image,  (224, 224))
    image_reshape = np.reshape(image_resize, [1, 224, 224, 3])
    prediction = model.predict(image_reshape)
    print(prediction)

read_image()