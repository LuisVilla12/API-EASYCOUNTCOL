from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt


def tratamiento_imagen(name_image):
    # Cargar la imagen tomando el arugmento de la línea de comandos
    image = cv2.imread(name_image)
    image_resultado = image.copy()  # Trabaja sobre una copia

    shifted = cv2.pyrMeanShiftFiltering(image_resultado, 21, 51)

    # Escala de grises
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    #Imbrusal 
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Calcular distancia euclidiana
    D = ndimage.distance_transform_edt(thresh)

    # Buscar máximos locales
    coordinates = peak_local_max(D, min_distance=20, labels=thresh)

    # Crear máscara booleana
    localMax = np.zeros_like(D, dtype=bool)
    localMax[tuple(coordinates.T)] = True

    # Etiquetar marcadores
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]

    # Aplicar Watershed
    labels = watershed(-D, markers, mask=thresh)

    # Mostrar cantidad detectada
    # print(f"[INFO] {len(np.unique(labels)) - 1} colonias detectadas")
    
    # Dibujar resultados
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image_resultado, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image_resultado, f"#{label}", (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return {
        "image_resultado": image_resultado,
        "labels": len(np.unique(labels)) - 1
        }


# resultados=tratamiento_imagen('example.jpg')
# print(type(resultados["image_resultado"]))
# print(type(resultados["labels"]))
# print(type(resultados["processing_time"]))