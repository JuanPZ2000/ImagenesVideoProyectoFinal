import dlib
import cv2
import numpy as np
import os
import sys
import pickle
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
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

#vid = cv2.VideoCapture(0)

#while True:

#ret, frame = vid.read()
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
    elif i>=50 and i<101:
        etiquetas.append(1) # tiste
    elif i >= 100 and i < 151:
        etiquetas.append(2)  # sorpesa
    else:
        etiquetas.append(3)  # sorpesa

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
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break
    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)
    # AQUI ACABA EL GUAIL
# After the loop release the cap object
#vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
with open("obj.pickle", "wb") as f:
    pickle.dump(distancia, f)
print("Hecho")