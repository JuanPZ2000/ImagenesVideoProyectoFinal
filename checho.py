import dlib
import cv2
import numpy as np
import os
import sys
import pickle

def lectura(path):
    data_dir_list = os.listdir(path)
    img_data_list = []
    for dataset in data_dir_list:
        img_list = os.listdir(path + '/' + dataset)
        for img in img_list:
            input_img = cv2.imread(path + '/' + dataset + '/' + img)
            # input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_resize = cv2.resize(input_img, (48, 48))
            img_data_list.append(input_img_resize)
    return img_data_list

predictor_path = os.path.abspath(os.getcwd()) + '\shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
distancia = []
etiquetas = []
path = os.path.abspath(os.getcwd())+'\ckplus'
images=lectura(path)
N=(len(images))
#cv2.imshow('a',images[731])
#cv2.waitKey(0)

for i in range(441,731):
    frame = images[i]

    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(new_gray, 1)
    distanciaaux = []

    # Etiquetas
    if i < 647:
        etiquetas.append(0) # Feli
    elif i>=647 and i<732:
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
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break
    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)
    # AQUI ACABA EL WHILE
# After the loop release the cap object
#vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
with open("obj.pickle", "wb") as f:
    pickle.dump(distancia, f)
print("Hecho")


"""
path = os.path.abspath(os.getcwd())+'\ckplus\happy'
N = len(os.listdir(path))
images=lectura(path,N)
cv2.imshow('a',images[6])
cv2.waitKey(0)"""