# import the necessary packages
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt

# analizar los argumentos de nuestra línea de comandos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
# Carga la imagen y realiza un filtrado piramidal de desplazamiento medio
image = cv2.imread(args["image"])
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
# Convierte la imagen desplazada a escala de grises, luego aplica el umbral de Otsu  para segmentar el fondo del primer plano
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# calcular la distancia euclidiana exacta de cada píxel binario al píxel cero más cercano, luego encontrar picos en este mapa de distancias
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)
# realizar la segmentación de watershed utilizando los picos locales como marcadores
#  realizar un análisis de componentes conectados en los picos locales, utilizando 8-conectividad, a continuación, aplicar el algoritmo Watershed

markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

plt.imshow(thresh)
plt.title("Thresh")
plt.axis("off")
plt.show()
plt.imshow(image)
plt.title("Input")
plt.axis("off")
plt.show()