import dlib
import cv2
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle



predictor_path = os.path.abspath(os.getcwd()) + '\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
path = os.path.abspath(os.getcwd()) + '\images'
N = len(os.listdir(path))

distancia = []
etiquetas = []

for i in range(N):
    if i < 50:
        etiquetas.append(0)  # Feli
    elif i >= 50 and i < 101:
        etiquetas.append(1)  # tiste
    elif i >= 100 and i < 151:
        etiquetas.append(2)  # sorpesa
    else:
        etiquetas.append(3)  # enojado

with open("obj.pickle", "rb") as f:
    distancia = pickle.load(f)
print(distancia)

#########################################################################################

# ML
X_train, X_test, y_train, y_test = train_test_split(distancia, etiquetas,test_size=0.2, random_state=0)
# Se escalan las distancias
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

k_range = range(1, int(np.sqrt(len(y_train))))
distance='minkowski'
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k,weights='distance',metric=distance, metric_params=None,algorithm='brute')
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_test)
    print(y_test,y_predicted)

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(new_gray, 1)
    dist=[]
    try:
        if dets[0] is not None:
            shape = predictor(new_gray, dets[0])
            old_points = np.zeros((68, 2), dtype=np.int32)
            for i in range(0, 68):
                old_points[i] = (shape.part(i).x, shape.part(i).y)
            for (x, y) in old_points:
                dist.append(((x - old_points[0][0]) * 2 + (y - old_points[0][1]) * 2) ** 1/2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            arreglo=np.array(dist)
            a=scaler.transform(arreglo.reshape(1,arreglo.shape[0]))
            y_predict=knn.predict(a)
            if(y_predict==0):
                print("feliz")
                cv2.putText(frame, "feliz", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,255), 5, bottomLeftOrigin=False)
            elif(y_predict==1):
                cv2.putText(frame, "triste", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5,
                            bottomLeftOrigin=False)
                print("triste")
            elif (y_predict == 2):
                cv2.putText(frame, "sorpresa", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 5,
                            bottomLeftOrigin=False)
                print("sorpresa")
            elif (y_predict == 3):
                cv2.putText(frame, "enojado", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5,
                            bottomLeftOrigin=False)
                print("sorpresa")
    except IndexError:
        pass
    cv2.imshow('Video', frame)
    cv2.waitKey(1)