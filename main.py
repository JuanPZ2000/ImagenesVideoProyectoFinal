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

img_feliz = cv2.imread("Resources/img_feliz.png", cv2.IMREAD_UNCHANGED)
img_triste = cv2.imread("Resources/img_triste.png", cv2.IMREAD_UNCHANGED)

scale_percent = 45

width = int(img_feliz.shape[1] * scale_percent / 100)
height = int(img_feliz.shape[0] * scale_percent / 100)

img_feliz = cv2.resize(img_feliz, (width, height))

width = int(img_triste.shape[1] * scale_percent / 100)
height = int(img_triste.shape[0] * scale_percent / 100)

img_triste = cv2.resize(img_triste, (width, height))

predictor_path = os.path.abspath(os.getcwd()) + "\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
path = os.path.abspath(os.getcwd()) + "\images"
N = len(os.listdir(path))

lst_of_lst_distancia = []
etiquetas = []
puntos_interes = range(17, 66)
for i in range(100):
    if i < 50:
        etiquetas.append(0)  # Feli
    elif i >= 50 and i < 101:
        etiquetas.append(1)  # tiste
    # elif i >= 100 and i < 151:
    #     etiquetas.append(2)  # sorpesa
    # else:
    #     etiquetas.append(3)  # enojado

for contador in range(1, 100 + 1):
    lst_distancia = []
    frame = cv2.imread("./images2/image" + str(contador) + ".jpeg")
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(new_gray, 1)
    try:
        if dets[0] is not None:
            shape = predictor(new_gray, dets[0])
            old_points = np.zeros((49, 2), dtype=np.int32)
            for i in range(0, 49):
                old_points[i] = (shape.part(i + 17).x, shape.part(i + 17).y)
            for (x, y) in old_points:
                lst_distancia.append(
                    ((x - old_points[16][0]) * 2 + (y - old_points[16][1]) * 2) ** 1 / 2
                )
            lst_of_lst_distancia.append(lst_distancia)
    except IndexError:
        etiquetas.pop(contador - 1)
# with open("obj.pickle", "rb") as f:
#     distancia = pickle.load(f)
# print(distancia)

#########################################################################################

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
while True:
    _, frame = cap.read()
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
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            arreglo = np.array(dist)
            a = scaler.transform(arreglo.reshape(1, arreglo.shape[0]))
            y_predict = knn.predict(a)
            if y_predict == 0:
                print("feliz")
                frame = cvzone.overlayPNG(
                    frame, img_feliz, [points[1][0] - 50, points[1][1] - 50]
                )
            elif y_predict == 1:
                frame = cvzone.overlayPNG(
                    frame, img_triste, [points[1][0] - 50, points[1][1] - 50]
                )
                print("triste")
    except:
        pass
    cv2.imshow("Video", frame)
    cv2.waitKey(1)
