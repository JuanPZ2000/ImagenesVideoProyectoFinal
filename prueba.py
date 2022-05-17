import cv2
import dlib
import os
import numpy as np

predictor_path = os.path.abspath(os.getcwd()) + "\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
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
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            scale_percent = 20  # percent of original size
            width = int(frame_copy.shape[1] * scale_percent / 100)
            height = int(frame_copy.shape[0] * scale_percent / 100)
            dim = (width, height)
            frame_copy = cv2.resize(frame_copy, dim, interpolation=cv2.INTER_AREA)
            frame[
                frame.shape[0] - frame_copy.shape[0] - 1 : frame.shape[0] - 1,
                frame.shape[1] - frame_copy.shape[1] - 1 : frame.shape[1] - 1,
                :,
            ] = frame_copy
            a = 1
    except:
        pass
    cv2.imshow("Video", frame)
    cv2.waitKey(1)
