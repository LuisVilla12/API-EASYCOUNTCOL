from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Colocar la ruta de la imagen")
args = vars(ap.parse_args())

# Cargar la imagen tomando el argumento de la línea de comandos
image = cv2.imread(args["image"])
if image is None:
    print("Error: no se pudo abrir la imagen:", args["image"])
    sys.exit(1)
image_resultado = image.copy()  # Trabaja sobre una copia

# Filtro Gaussiano (reduce ruido fino)
imagen_suavizada = cv2.GaussianBlur(image, (5, 5), 7)

# Filtro Mean Shift (suaviza regiones preservando bordes)
shifted = cv2.pyrMeanShiftFiltering(imagen_suavizada, 30, 30)

# Escala de grises
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

# Aplicar umbral Otsu
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Operación morfológica para limpiar ruido sin eliminar microcolonias
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Detectar automáticamente el círculo de la caja Petri (más tolerante)
circles = cv2.HoughCircles(
    gray, cv2.HOUGH_GRADIENT, dp=1.2,
    minDist=max(20, gray.shape[0] // 8),
    param1=50, param2=30,
    minRadius=int(0.18 * gray.shape[0]), maxRadius=int(0.48 * gray.shape[0])
)

mask_circular = np.zeros(gray.shape, dtype="uint8")
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    c = max(circles, key=lambda x: x[2])
    radio_detectado = c[2]
    centro = (c[0], c[1])

    # Prueba márgenes crecientes hasta que la franja del borde tenga pocos píxeles blancos
    umbral_borde = 0.05  # Porcentaje máximo de píxeles blancos permitidos en la franja
    ancho_franja = max(8, int(0.03 * radio_detectado))    # ancho relativo
    margen = 0
    while margen < int(0.1 * radio_detectado):
        radio_interno = int(radio_detectado - margen - ancho_franja)
        radio_externo = int(radio_detectado - margen)
        mascara_franja = np.zeros(gray.shape, dtype="uint8")
        cv2.circle(mascara_franja, centro, radio_externo, 255, -1)
        cv2.circle(mascara_franja, centro, max(0, radio_interno), 0, -1)
        pixeles_borde = cv2.bitwise_and(thresh, thresh, mask=mascara_franja)
        porcentaje_blancos = np.sum(pixeles_borde > 0) / (np.sum(mascara_franja > 0) + 1e-8)
        if porcentaje_blancos < umbral_borde:
            break
        margen += 2

    radio_mascara = int(radio_detectado - margen)
    mask_circular = np.zeros(gray.shape, dtype="uint8")
    cv2.circle(mask_circular, centro, radio_mascara, 255, -1)
    print(f"Margen automático aplicado: {margen} píxeles (radio final: {radio_mascara})")

    # Erosión suave y consistente
    erosion_px = max(1, int(round(0.0025 * radio_mascara)))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px * 2 + 1, erosion_px * 2 + 1))
    mask_circular_eroded = cv2.erode(mask_circular, kernel_ellipse, iterations=1)
else:
    # Si no se detecta círculo, usar el centro y radio por defecto (consistente con erosion_px)
    centro = (gray.shape[1] // 2, gray.shape[0] // 2)
    radio_mascara = min(centro) - 10
    cv2.circle(mask_circular, centro, radio_mascara, 255, -1)
    print(f"No se detectó círculo, usando centro=({centro[0]}, {centro[1]}), radio={radio_mascara}")

    erosion_px = max(1, int(round(0.0025 * radio_mascara)))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px * 2 + 1, erosion_px * 2 + 1))
    mask_circular_eroded = cv2.erode(mask_circular, kernel_ellipse, iterations=1)

# Construye la franja de borde y aplica la máscara erosionada UNA vez
mask_edge = cv2.bitwise_xor(mask_circular, mask_circular_eroded)
thresh = cv2.bitwise_and(thresh, thresh, mask=mask_circular_eroded)
# Cerrar pequeños huecos para recuperar microcolonias pegadas al borde
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)

# Mapa de distancia y máximos locales
D = ndimage.distance_transform_edt(thresh)

# Umbrales relativos para distancia y separación
min_distancia = max(6, int(0.02 * radio_mascara))
umbral_distancia = max(6, int(0.12 * D.max()))

coordinates = peak_local_max(D, min_distance=min_distancia, labels=thresh)

# Filtrar máximos por umbral relativo del mapa de distancia
coordinates_filtradas = []
valores_umbral = []
for coord in coordinates:
    valor = D[coord[0], coord[1]]
    if valor >= umbral_distancia:
        coordinates_filtradas.append(coord)
        valores_umbral.append(valor)
coordinates_filtradas = np.array(coordinates_filtradas)

# Informar valores (útil para ajustar parámetros)
print("Valores del mapa de distancia en cada máximo local filtrado:")
for i, valor in enumerate(valores_umbral):
    print(f"Punto {i+1}: {valor:.2f}")

localMax = np.zeros_like(D, dtype=bool)
if len(coordinates_filtradas) > 0:
    localMax[tuple(coordinates_filtradas.T)] = True

# Etiquetar marcadores y aplicar Watershed
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)

contador_colonias = 0

for label in np.unique(labels):
    if label == 0:
        continue
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        continue

    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    ((x, y), r) = cv2.minEnclosingCircle(c)

    # FILTRADO MEJORADO (parámetros coherentes y relativos)
    radio_mascara_efectivo = radio_mascara - erosion_px
    margen_borde = max(int(0.06 * radio_mascara_efectivo), 10)

    distancia_al_centro = np.sqrt((x - centro[0]) ** 2 + (y - centro[1]) ** 2)

    # filtros por área mínima y circularidad
    min_area = max(20, int(np.pi * (2.0 ** 2)))
    perim = cv2.arcLength(c, True)
    circularity = 0
    if perim > 0:
        circularity = 4 * np.pi * area / (perim * perim)

    borde_tol = max(4, int(0.02 * radio_mascara))

    # Solapamiento con la franja
    area_contour_px = np.count_nonzero(mask > 0)
    if area_contour_px == 0:
        continue
    area_overlap_px = np.count_nonzero(np.logical_and(mask_edge > 0, mask > 0))
    overlap_ratio = (area_overlap_px / area_contour_px)

    # Área dentro de la máscara original y dentro de la erosionada
    area_inside_original = np.count_nonzero(np.logical_and(mask_circular > 0, mask > 0))
    area_inside_eroded = np.count_nonzero(np.logical_and(mask_circular_eroded > 0, mask > 0))
    frac_inside_original = area_inside_original / area_contour_px
    frac_inside_eroded = area_inside_eroded / area_contour_px

    # Descartar si completamente fuera
    if area_inside_original == 0:
        continue

    # Umbrales ajustables (más restrictivos en presencia de solapamiento con el aro)
    overlap_threshold = 0.40
    min_frac_inside_original = 0.90
    # Aumentado para evitar contar trozos del aro
    min_frac_inside_eroded = 0.50

    # Si solapa mucho y no hay suficiente parte dentro, descartar
    if overlap_ratio > overlap_threshold and not (frac_inside_original >= min_frac_inside_original or frac_inside_eroded >= min_frac_inside_eroded):
        continue

    # Comprobación geometrica estricta: el centro del contorno debe estar dentro de la máscara erosionada
    M = cv2.moments(c)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # sin momento fiable, usar minEnclosingCircle center
        cx, cy = int(x), int(y)

    # Si el centro está fuera de la máscara erosionada y la fracción dentro es baja, descartar
    if mask_circular_eroded[cy, cx] == 0 and frac_inside_eroded < min_frac_inside_eroded:
        continue

    # Evitar objetos que sobresalen demasiado del radio interno efectivo
    if (distancia_al_centro + r) > (radio_mascara_efectivo - max(3, int(0.01 * radio_mascara))):
        # si aún así hay mucha área dentro de la máscara original, permitir; si no, descartar
        if frac_inside_original < min_frac_inside_original:
            continue

    # Comprobaciones adicionales de área y circularidad
    if area < min_area or circularity < 0.22:
        continue

    # Aceptar como colonia válida
    contador_colonias += 1

    # Ajustar dibujo para no pintar sobre el aro externo
    orig_r = max(1, int(round(r)))
    draw_r = max(1, orig_r - 3)
    if overlap_ratio > 0.05:
        extra_reduc = int(min(orig_r * 0.6, max(2, overlap_ratio * orig_r * 2)))
        draw_r = max(1, orig_r - extra_reduc)

    cv2.circle(image_resultado, (int(x), int(y)), draw_r, (255, 255, 255), 2)
    txt_x = int(x) - min(10, draw_r)
    txt_y = int(y)
    cv2.putText(image_resultado, f"#{contador_colonias}", (txt_x, txt_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

print(f"[INFO] {contador_colonias} colonias detectadas")

# Mostrar con Matplotlib
plt.figure(figsize=(14, 7))
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Imagen original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(cv2.cvtColor(image_resultado, cv2.COLOR_BGR2RGB))
plt.title("Colonias detectadas")
plt.axis("off")

plt.tight_layout()
plt.show()
