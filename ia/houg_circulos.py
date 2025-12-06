import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser() 
ap.add_argument("-i", "--image", required=True, help="Colocar la ruta de la imagen") 
args = vars(ap.parse_args())


# Cargar imagen desde los argumentos
img = cv2.imread(args["image"])

# --- Preprocesamiento ---
suavizada = cv2.GaussianBlur(img, (5, 5), 0)
gray = cv2.cvtColor(suavizada, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 3
)

# --- Detección de círculos con Hough ---
circles = cv2.HoughCircles(
    thresh,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=20,
    param1=40,
    param2=40,
    minRadius=20,
    maxRadius=500
)

# Copia para dibujar los resultados
output = img.copy()

# --- Dibujar y contar círculos detectados ---
if circles is not None:
    circles = np.uint16(np.around(circles))
    conteo = circles.shape[1]

    for (x, y, r) in circles[0, :]:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)   # borde del círculo
        cv2.circle(output, (x, y), 2, (255, 0, 0), 3)   # centro

    print(f"Colonias detectadas: {conteo}")
else:
    print("No se detectaron círculos.")

# --- Mostrar con Matplotlib ---
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Imagen original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(suavizada, cv2.COLOR_BGR2RGB))
plt.title("Imagen suavizada")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(thresh, cmap="gray")
plt.title("Trehsholding")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(gray, cmap="gray")
plt.title("Gris")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Colonias detectadas")
plt.axis("off")

plt.tight_layout()
plt.show()
