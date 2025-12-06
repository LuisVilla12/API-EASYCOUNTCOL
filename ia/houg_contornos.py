import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

ap = argparse.ArgumentParser() 
ap.add_argument("-i", "--image", required=True, help="Ruta de la imagen de la placa") 
args = vars(ap.parse_args())

# --- Cargar imagen ---
img = cv2.imread(args["image"])

# --- Preprocesamiento ---
suavizada = cv2.GaussianBlur(img, (5, 5), 0)
gray = cv2.cvtColor(suavizada, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

# Umbral adaptativo
thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 3
)

# --- Crear máscara circular para ignorar fuera del plato ---
mask = np.zeros_like(gray)
cv2.circle(mask, (gray.shape[1]//2, gray.shape[0]//2), gray.shape[0]//2 - 10, 255, -1)
thresh = cv2.bitwise_and(thresh, mask)

# --- Limpieza de ruido (operaciones morfológicas) ---
kernel = np.ones((3,3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# --- Detección de contornos ---
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar colonias por tamaño (área mínima y máxima)
colonias = [c for c in contours if 50 < cv2.contourArea(c) < 2000]
# conteo = len(colonias)

# --- Dibujar colonias detectadas ---
output = img.copy()
for c in colonias:
    (x, y), r = cv2.minEnclosingCircle(c)
    cv2.circle(output, (int(x), int(y)), int(r), (0, 255, 0), 2)

print(f"Colonias detectadas: {conteo}")

# --- Mostrar resultados ---
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
plt.imshow(gray, cmap="gray")
plt.title("Gris")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(thresh, cmap="gray")
plt.title("Threshold + máscara")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Colonias detectadas")
plt.axis("off")

plt.tight_layout()
plt.show()
