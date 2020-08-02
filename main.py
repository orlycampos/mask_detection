import cv2
import mtcnn
import numpy as np
from keras.models import load_model

model = load_model("/Users/camposo/Desktop/Camera Testing/model_ex-002_acc-0.703598.h5")

def start_camera():
    #cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        #cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        detector = mtcnn.MTCNN()
        result = detector.detect_faces(frame)
        if len(result) > 0:
            image = crop_image(result, frame)
            image_resize = cv2.resize(image, (224, 224))
            image_reshape = np.reshape(image_resize, [1, 224, 224, 3])
            prediction = model.predict(image_reshape)
            #print(prediction)
        if key == 27: # exit on ESC
            break
    #cv2.destroyWindow("preview")

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
        #image_cropped = image[bounding_box[0]: bounding_box[1],
        #              bounding_box[0] + bounding_box[2]: bounding_box[1] + bounding_box[3]]

        #image_cropped = image[bounding_box[0]: bounding_box[0] + bounding_box[2],
        #                bounding_box[1]: bounding_box[1] + bounding_box[3]]
        image_cropped = image[bounding_box[1]: bounding_box[1] + bounding_box[3],
                        bounding_box[0]: bounding_box[0] + bounding_box[2]]
        cv2.imshow('image', image_cropped)
        return image_cropped

def crop_rect_new(img, rect):
    # get the parameter of the small rectangle
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    out = cv2.getRectSubPix(img, size, center)
    return out

def get_rect_faces(image, face_coord):
    for obj in face_coord:
        cnt = get_rect_coordinates(obj)
        rect = cv2.minAreaRect(cnt)
        print("rect: {}".format(rect))
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        print("bounding box: {}".format(box))
        im_crop = crop_rect_new(image, rect)
        #cv2_imshow(im_crop)
        return im_crop

def get_rect_coordinates(coord):
    for key, val in coord.items():
        if key == "box":
            coord = np.array([[[val[0], val[1]]],
                              [[val[0] + val[2], val[1]]],
                              [[val[0], val[1] + val[3]]],
                              [[val[0] + val[2], val[1] + val[2]]]])
    return coord

start_camera()
