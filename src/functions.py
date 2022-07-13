import dlib
import cvzone
import cv2
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle


def deteccion_rostro(frame, detector, predictor):
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(new_gray, 1)
    points = []
    if len(dets) != 0:
        if dets[0] is not None:
            shape = predictor(new_gray, dets[0])
            points = np.zeros((49, 2), dtype=np.int32)
            for i in range(0, 49):
                points[i] = (shape.part(i + 17).x, shape.part(i + 17).y)
            flag_face = True
    else:
        shape = []
        flag_face = False
    return [points, shape, flag_face]


def predict(knn, dist, scaler, points, frame_copy_draw, img_feliz, img_triste):
    arreglo = np.array(dist)
    predict_item = scaler.transform(arreglo.reshape(1, arreglo.shape[0]))
    y_predict = knn.predict(predict_item)
    if y_predict == 0:
        # print(contador)
        # if (timer_1 - timer_2) > 0.001:

        #     contador = contador+5
        #     if contador == 100:
        #         contador = 70
        #     rojo.ChangeDutyCycle(100 - contador)
        #     timer_2 = timer_1
        # else:
        #     timer_1 = time.perf_counter()
        frame_copy_draw = cvzone.overlayPNG(
            frame_copy_draw, img_feliz, [points[1][0] - 25, points[1][1] - 30]
        )
    elif y_predict == 1:
        frame_copy_draw = cvzone.overlayPNG(
            frame_copy_draw, img_triste, [points[1][0] - 25, points[1][1] - 30]
        )
    return frame_copy_draw
    # rojo.ChangeDutyCycle(0)
