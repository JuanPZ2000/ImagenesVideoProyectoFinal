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
import wiringpi
import time
import RPi.GPIO as GPIO

    

wiringpi.wiringPiSetup()
wiringpi.pinMode(12, 1)

img_feliz = cv2.imread("Resources/img_feliz.png", cv2.IMREAD_UNCHANGED)
img_triste = cv2.imread("Resources/img_triste.png", cv2.IMREAD_UNCHANGED)

scale_percent = 22
width = int(img_feliz.shape[1] * scale_percent / 100)
height = int(img_feliz.shape[0] * scale_percent / 100)

img_feliz = cv2.resize(img_feliz, (width, height))

width = int(img_triste.shape[1] * scale_percent / 100)
height = int(img_triste.shape[0] * scale_percent / 100)

img_triste = cv2.resize(img_triste, (width, height))

predictor_path = os.path.abspath(os.getcwd()) + "/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
path = os.path.abspath(os.getcwd()) + "/images"
N = len(os.listdir(path))

etiquetas = []
puntos_interes = range(17, 66)
for i in range(100):
    if i < 50:
        etiquetas.append(0)  # Feli
    elif i >= 50 and i < 101:
        etiquetas.append(1)  # tiste

#########################################################################################
lst_of_lst_distancia = np.load("distancias.npy")
# ML
X_train, X_test, y_train, y_test = train_test_split(
    lst_of_lst_distancia, etiquetas, test_size=0.2, random_state=0
)
# Se escalan las distancias
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

k_range = range(1, int(np.sqrt(len(y_train))))
distance = "minkowski"
for k in k_range:
    knn = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance",
        metric=distance,
        metric_params=None,
        algorithm="brute",
    )
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_test)
    print(y_test, y_predicted)

cap = cv2.VideoCapture(0)
GPIO.setmode(GPIO.BCM)
GPIO.setup(24, GPIO.OUT)
rojo = GPIO.PWM(24, 100) 
rojo.start(100)
rojo.ChangeDutyCycle(100)
i=0
a=0
while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (320, 240))
    frame_copy = frame.copy()
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(new_gray, 1)
    dist = []
    try:
        if dets[0] is not None:
            shape = predictor(new_gray, dets[0])
            points = np.zeros((49, 2), dtype=np.int32)
            for i in range(0, 49):
                points[i] = (shape.part(i + 17).x, shape.part(i + 17).y)
            for (x, y) in points:
                dist.append(
                    ((x - points[16][0]) * 2 + (y - points[16][1]) * 2) ** 1 / 2
                )
                #cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            scale_percent = 20

            # Se pone la imagen original debajo
            width = int(frame_copy.shape[1] * scale_percent / 100)
            height = int(frame_copy.shape[0] * scale_percent / 100)
            dim = (width, height)
            frame_copy = cv2.resize(frame_copy, dim, interpolation=cv2.INTER_AREA)
            frame[
                frame.shape[0] - frame_copy.shape[0] - 1 : frame.shape[0] - 1,
                frame.shape[1] - frame_copy.shape[1] - 1 : frame.shape[1] - 1,
                :,
            ] = frame_copy

            # Se toman las distancias de los puntos para realizar la prediccion
            arreglo = np.array(dist)
            a = scaler.transform(arreglo.reshape(1, arreglo.shape[0]))
            y_predict = knn.predict(a)
            if y_predict == 0:
                for i in range(100,-1,-1):
                    rojo.ChangeDutyCycle(100 - i)
                    time.sleep(0.02)
                frame = cvzone.overlayPNG(
                    frame, img_feliz, [points[1][0] - 25, points[1][1] - 30]
                )
            elif y_predict == 1:
                frame = cvzone.overlayPNG(
                    frame, img_triste, [points[1][0] - 25, points[1][1] - 30]
                )
                rojo.ChangeDutyCycle(0)
                
                
    except:
        pass
    cv2.imshow("Video", cv2.resize(frame, (1280, 720)))
    cv2.waitKey(5)
