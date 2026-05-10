
import numpy as np
import math
import cv2 as cv
import socket
import json
import matplotlib.pyplot as plt

config = {
    # Paramètres intrinsèques webcam
    "sensor_mm": np.array([3.58, 2.685]),
    "focal_mm": 4,
    "resolution": np.array([1280, 960]),

    # Webcam
    "id_cam1": 4,
}


resolution = config["resolution"]
center = (resolution[0] / 2, resolution[1] / 2)
focal_mm = config["focal_mm"]
sensor_mm = config["sensor_mm"]




cap1 = cv.VideoCapture(config["id_cam1"])

resolution = config["resolution"]
cap1.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
cap1.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])

def ball_detection(frame):
    ball_detected = False
    # Convertir l'image en espace de couleur HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_or = frame.copy()
    # Définir les bornes pour la couleur jaune (typique d'une balle de tennis)
    lower_yellow = np.array([0, 100, 100])  # Plage de couleurs jaunes (en HSV)
    upper_yellow = np.array([50, 255, 255])

    # Créer un masque pour extraire la couleur jaune (balle de tennis)
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    # Appliquer le masque à l'image originale
    yellow_objects = cv.bitwise_and(frame, frame, mask=mask)

    # Convertir l'image masquée en niveaux de gris
    gray = cv.cvtColor(yellow_objects, cv.COLOR_BGR2GRAY)

    threshold = 100
    gray[gray < threshold] = 0

    # Appliquer un flou pour aider à la détection des cercles
    gray_blurred = cv.GaussianBlur(gray, (15, 15), 0)


    # Détecter les cercles avec la méthode de Hough
    circles = cv.HoughCircles(
        gray_blurred, 
        cv.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=100
    )
    if cv.waitKey(1) & 0xFF == ord('s'):
        cv.imwrite('captured_image.png', gray_blurred)
        print("Image capturée et sauvegardée.")
    # Vérifier si des cercles ont été détectés
    if circles is not None:
        ball_detected = True
        circles = np.round(circles[0, :]).astype("int")  # Convertir les coordonnées en entiers
        largest_circle = circles[0]
        largest_rayon = circles[0][2]
        i = 1
        for i in range(len(circles)):
            if circles[i][2] > largest_rayon:
                largest_rayon = circles[i][2]
                largest_circle = circles[i]

        # Dessiner les cercles détectés sur l'image originale
        (x, y, r) = largest_circle
        cv.circle(frame, (x, y), r, (0, 255, 0), 4)  # Dessiner le cercle en vert
        cv.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # Marquer le centre en orange
        cv.imshow('Image Originale', frame_or)
        cv.imshow('Gray Image', gray_blurred)
        cv.imshow('Ball Detection', frame)

    
        return ball_detected
    else:
        return ball_detected

    
if __name__ == "__main__":
    try:
        while cap1.isOpened():
            ret, frame = cap1.read()
            ball_detection(frame)
            #cv.imshow('Ball Detection', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap1.release()
        cv.destroyAllWindows()