import dlib
import cv2
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def lectura(path, N):
    images = []
    for i in range(N):
        image_name = "image" + str(i + 1) + ".jpeg"
        path_file = os.path.join(path, image_name)
        images.append(cv2.imread(path_file))
    return images

predictor_path = os.path.abspath(os.getcwd()) + '\shape_predictor_68_face_landmarks.dat'
#predictor_path = 'C:/Users/usuario/Downloads/shape/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


path = os.path.abspath(os.getcwd()) + '\images'
N = len(os.listdir(path))
images = lectura(path, N)
distancia = []
etiquetas = []

for i in range(N):
    frame = images[i]

    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(new_gray, 1)
    distanciaaux = []

    # Etiquetas
    if i < 50:
        etiquetas.append(0) # Feli
    else:
        etiquetas.append(1) # tiste

    try:
        if dets[0] is not None:
            shape = predictor(new_gray, dets[0])
            old_points = np.zeros((68, 2), dtype=np.int32)
            for i in range(0, 68):
                old_points[i] = (shape.part(i).x, shape.part(i).y)
            for (x, y) in old_points:
                distanciaaux.append(((x - old_points[0][0]) * 2 + (y - old_points[0][1]) * 2) ** 1/2)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            distancia.append(distanciaaux)

    except IndexError:
        distancia.append(list(np.zeros(68)))
        pass

cv2.destroyAllWindows()

#########################################################################################

# ML
X_train, X_test, y_train, y_test = train_test_split(distancia, etiquetas,test_size=0.1, random_state=0)
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
            elif(y_predict==1):
                print("triste")
    except IndexError:
        pass
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
cap.release()