from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Colocar la ruta de la imagen")
args = vars(ap.parse_args())

# Cargar la imagen tomando el arugmento de la línea de comandos
image = cv2.imread(args["image"])
image_resultado = image.copy()  # Trabaja sobre una copia

# Filtro Gaussiano (reduce ruido fino)
imagen_suavizada=cv2.GaussianBlur(image,(5,5,),7)

# Filtro Mean Shift (suaviza regiones preservando bordes)
shifted = cv2.pyrMeanShiftFiltering(imagen_suavizada, 30,30)

# Escala de grises
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)

# Aplicar la máscara al umbral
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Crea una matriz kernel que define el área con la que se realizarán las operaciones morfológicas.
kernel = np.ones((7, 7), np.uint8)  # (3x3 menos agresivo,5x5, 7x7 más agresivo)
# Aplica apertura para eliminar ruido pequeño
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# Transformada de Hough para detectar el borde circular de una caja de Petri
circles = cv2.HoughCircles(
    # Imagen en escala de grises
    gray, 
    # Método de detección (Gradiente de Hough)
    cv2.HOUGH_GRADIENT, 
    # Método de detección (Gradiente de Hough)
    dp=1.2, 
    # Distancia mínima entre centros de círculos
    minDist=gray.shape[0]//2,
    # Umbral superior para Canny
    param1=50, 
    # Umbral para el acumulador de centros
    param2=30, 
    # Radio mínimo esperado
    minRadius=gray.shape[0]//4, 
    # Radio máximo esperado
    maxRadius=gray.shape[0]//2
)
# # Define una mascara  una máscara binaria vacía  mismo tamaño que la imagen en escala de grises gray.
mask_circular = np.zeros(gray.shape, dtype="uint8")

# Si se detecta un círculo, usarlo para crear una máscara circular
if circles is not None:
    # Detecta el círculo más grande (asumiendo que es la caja de Petri)
    circles = np.round(circles[0, :]).astype("int")
    c = max(circles, key=lambda x: x[2])
    # Almacena el radio y centro dectado
    radio_detectado = c[2]
    centro = (c[0], c[1])

    # Ajuste dinámico del margen para la máscara circular
    umbral_borde = 0.05  # Porcentaje máximo de píxeles blancos permitidos en la franja
    ancho_franja = 5    # Ancho de la franja en píxeles
    margen = 0
    
    # Crea una franja anular alrededor del borde, qué porcentaje de píxeles blancos hay en esa zona. Si hay mucho ruido, reduce progresivamente el área  (incrementando el margen) hasta que el borde esté limpio.
    
    while margen < int(0.12 * radio_detectado):  # No restar más del 12% del radio
        radio_interno = int(radio_detectado - margen - ancho_franja)
        radio_externo = int(radio_detectado - margen)
        # Crear máscara para la franja circular
        mascara_franja = np.zeros(gray.shape, dtype="uint8")
        cv2.circle(mascara_franja, centro, radio_externo, 255, -1)
        cv2.circle(mascara_franja, centro, radio_interno, 0, -1)
        
        # Aplica la franja a la imagen umbralizada
        pixeles_borde = cv2.bitwise_and(thresh, thresh, mask=mascara_franja)
        porcentaje_blancos = np.sum(pixeles_borde > 0) / np.sum(mascara_franja > 0)
        if porcentaje_blancos < umbral_borde:
            break
        margen += 2  # Aumenta el margen de a 2 píxeles

    # Genera la máscara binaria para el  interior de la caja Petri.
    radio_mascara = int(radio_detectado - margen)
    mask_circular = np.zeros(gray.shape, dtype="uint8")
    cv2.circle(mask_circular, centro, radio_mascara, 255, -1)
    print(f"Margen automático aplicado: {margen} píxeles (radio final: {radio_mascara})")

    # Erosiona suavemente para eliminar artefactos pegados al borde sin perder colonias reales cercanas.
    erosion_px = max(1, int(0.003 * radio_mascara))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px*2+1, erosion_px*2+1))
    mask_circular_eroded = cv2.erode(mask_circular, kernel_ellipse, iterations=1)

    # Usa la máscara erosionada para el umbral final (evita falsos positivos en el borde)
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask_circular_eroded)
    
    # Cerrar pequeños huecos para recuperar microcolonias pegadas al borde
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Limitar el umbral al área interior de la caja Petri
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)
else:
    # Si no se detecta círculo, usar el centro y radio por defecto
    center = (gray.shape[1] // 2, gray.shape[0] // 2)
    radius = min(center) - 10
    cv2.circle(mask_circular, center, radius, 255, -1)
    print(f"No se detectó círculo, usando centro=({center[0]}, {center[1]}), radio={radius}")

    centro = center
    radio_mascara = radius
    erosion_px = max(3, int(0.03 * radio_mascara))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_px*2+1, erosion_px*2+1))
    mask_circular_eroded = cv2.erode(mask_circular, kernel_ellipse, iterations=1)

# Crear máscara de la franja del borde para análisis posterior
mask_edge = cv2.bitwise_xor(mask_circular, mask_circular_eroded)
thresh = cv2.bitwise_and(thresh, thresh, mask=mask_circular_eroded)

# Cerrar pequeños huecos para recuperar microcolonias pegadas al borde
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=1)

# Conversión a otros espacios de color
# usa imagen suavizada para reducir ruido en la detección de reflejos
hsv = cv2.cvtColor(shifted, cv2.COLOR_BGR2HSV)
v = hsv[:, :, 2].astype(np.uint8)
lab = cv2.cvtColor(shifted, cv2.COLOR_BGR2LAB)
Lchan = lab[:, :, 0]

# Referencia de brillo dentro de la placa (si mask_circular válida)
mask_plate = (mask_circular > 0)
if np.count_nonzero(mask_plate) > 0:
    mean_v_plate = np.mean(v[mask_plate])
    std_v_plate = np.std(v[mask_plate])
else:
    mean_v_plate = np.mean(v)
    std_v_plate = np.std(v)

# detectar píxeles especulares en la franja: muy brillantes respecto a la placa
v_blur = cv2.GaussianBlur(v, (7, 7), 0)
# umbral adaptativo simple: media + k*std o un corte absoluto alto
specular_thresh = int(min(255, mean_v_plate + max(18, 1.6 * std_v_plate)))
reflection_mask = np.zeros_like(v_blur, dtype=np.uint8)
reflection_mask[(v_blur >= specular_thresh) & (mask_edge > 0)] = 255

# opcional: dilatar la máscara de reflejo para cubrir halo
kernel_spec = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
reflection_mask = cv2.dilate(reflection_mask, kernel_spec, iterations=1)

# Combina la máscara de reflejo con la máscara circular erosionada
mask_final = cv2.bitwise_or(mask_circular_eroded, reflection_mask)

# Calcula el valor máximo del mapa de distancia para cada marcador
min_distancia = 9  # Ajusta según separación mínima esperada

# umbral adaptativo más robusto
D = ndimage.distance_transform_edt(thresh)


# Valores dentro de la placa (no fondo)
nonzero_D = D[thresh > 0]
if nonzero_D.size > 0:
    # usa un factor más pequeño y limitar por un percentil del mapa D
    # factor_radio = 0.06            # antes 0.12, reducir para no filtrar demasiado
    # umbral_por_radio = int(factor_radio * radio_mascara)
    # umbral_por_percentil = int(np.percentile(nonzero_D, 75))  # 75º percentil
    # umbral_distancia = max(6, min(umbral_por_radio, umbral_por_percentil))
    umbral_distancia = 9
else:
    umbral_distancia = 9

# umbral_distancia = 9
coordinates = peak_local_max(D, min_distance=min_distancia, labels=thresh)

# Solo selecciona los máximos locales que superan el umbral
coordinates_filtradas = []
valores_umbral = []
for coord in coordinates:
    valor = D[coord[0], coord[1]]
    if valor > umbral_distancia:
        coordinates_filtradas.append(coord)
        valores_umbral.append(valor)
coordinates_filtradas = np.array(coordinates_filtradas)

# Imprime los valores
print("Valores del mapa de distancia en cada máximo local filtrado:")
for i, valor in enumerate(valores_umbral):
    print(f"Punto {i+1}: {valor}")
    
    
localMax = np.zeros_like(D, dtype=bool)
if len(coordinates_filtradas) > 0:
    localMax[tuple(coordinates_filtradas.T)] = True

# Etiquetar marcadores
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]

# Aplicar Watershed
labels = watershed(-D, markers, mask=thresh)

# Mostrar cantidad detectada
# print(f"[INFO] {len(np.unique(labels)) - 1} etiquetas dectectadas (sin contar fondo)")

contador_colonias = 0  # Nuevo contador
debug = True  # Poner False para silencio

def _reject(label, reason, **kw):
    if debug:
        info = " ".join(f"{k}={v}" for k, v in kw.items())
        print(f"[RECHAZADO] label={label} reason={reason} {info}")

for label in np.unique(labels):
    if label == 0:
        continue
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        ((x, y), r) = cv2.minEnclosingCircle(c)

        # FILTRADO MEJORADO
        radio_mascara_efectivo = radio_mascara - erosion_px
        margen_borde = max(int(0.06 * radio_mascara_efectivo), 12)

        distancia_al_centro = np.sqrt((x - centro[0])**2 + (y - centro[1])**2)

        # filtros por área mínima y circularidad
        min_area = max(20, int(np.pi * (2.0**2)))  # reducido para permitir microcolonias
        perim = cv2.arcLength(c, True)
        circularity = 0
        if perim > 0:
            circularity = 4 * np.pi * area / (perim * perim)

        borde_tol = max(5, int(0.02 * radio_mascara))

        # Solapamiento con la franja
        area_contour_px = np.count_nonzero(mask > 0)
        if area_contour_px == 0:
            _reject(label, "area_contour_px==0")
            continue
        area_overlap_px = np.count_nonzero(np.logical_and(mask_edge > 0, mask > 0))
        overlap_ratio = (area_overlap_px / area_contour_px)

        # brillo medio del contorno en V y en L
        mean_v_contour = float(np.mean(v[mask > 0])) if np.count_nonzero(mask) > 0 else 0.0
        mean_L_contour = float(np.mean(Lchan[mask > 0])) if np.count_nonzero(mask) > 0 else 0.0

        reflect_overlap_px = np.count_nonzero(np.logical_and(reflection_mask > 0, mask > 0))
        reflect_overlap_ratio = reflect_overlap_px / area_contour_px

        # Reglas para descartar reflexiones (más permisivas)
        if reflect_overlap_ratio > 0.60:
            _reject(label, "reflect_overlap_ratio alto", reflect_overlap_ratio=round(reflect_overlap_ratio, 2))
            continue
        if mean_v_contour >= (mean_v_plate + max(30, 4.0 * std_v_plate)):
            print("[DEBUG] mean_v_contour demasiado alto comparado con placa:")
            print(mean_v_plate, std_v_plate)
            _reject(label, "brillo muy alto (posible reflejo)", mean_v_contour=round(mean_v_contour,1), mean_v_plate=round(mean_v_plate,1))
            continue

        area_inside_original = np.count_nonzero(np.logical_and(mask_circular > 0, mask > 0))
        area_inside_eroded = np.count_nonzero(np.logical_and(mask_circular_eroded > 0, mask > 0))
        frac_inside_original = area_inside_original / area_contour_px
        frac_inside_eroded = area_inside_eroded / area_contour_px

        if area_inside_original == 0:
            _reject(label, "fuera de máscara circular original", area_inside_original=area_inside_original)
            continue

        # Umbrales ajustables: relajar aceptación si solapa con franja
        overlap_threshold = 0.80     # antes 0.55
        min_frac_inside_original = 0.15
        min_frac_inside_eroded = 0.45

        if overlap_ratio > overlap_threshold and not (frac_inside_original >= min_frac_inside_original or frac_inside_eroded >= min_frac_inside_eroded):
            _reject(label, "overlap_ratio alto y no dentro suficiente", overlap_ratio=round(overlap_ratio,2), frac_inside_eroded=round(frac_inside_eroded,2))
            continue

        if frac_inside_eroded < min_frac_inside_eroded:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if mask_circular_eroded[cy, cx] == 0:
                    _reject(label, "centro fuera de máscara erosionada", cx=cx, cy=cy)
                    continue
            else:
                _reject(label, "moments m00==0")
                continue

        if area < min_area or circularity < 0.03:
            _reject(label, "area o circularidad bajo", area=area, circularity=round(circularity,3))
            continue

        # Si pasa todos los filtros, contar
        contador_colonias += 1
        orig_r = max(1, int(round(r)))
        draw_r = max(1, orig_r - 3)
        if overlap_ratio > 0.05:
            extra_reduc = int(min(orig_r * 0.6, max(2, overlap_ratio * orig_r * 2)))
            draw_r = max(1, orig_r - extra_reduc)

        cv2.circle(image_resultado, (int(x), int(y)), draw_r, (0, 255, 0), 2)
        txt = f"#{contador_colonias}"
        txt_x = int(x) - min(10, draw_r)
        txt_y = int(y) + int(draw_r / 2)
        cv2.putText(image_resultado, txt, (txt_x, txt_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image_resultado, txt, (txt_x, txt_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

print(f"[INFO] {contador_colonias} colonias detectadas")

# Mostrar con Matplotlib
# plt.figure(figsize=(14, 7))
# plt.subplot(2, 4, 1)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("Imagen original")
# plt.axis("off")

# plt.subplot(2, 4, 2)
# plt.imshow(cv2.cvtColor(imagen_suavizada, cv2.COLOR_BGR2RGB))
# plt.title("imagen suavizada")
# plt.axis("off")

# plt.subplot(2, 4, 3)
# plt.imshow(gray, cmap="gray")
# plt.title("Gris")
# plt.axis("off")

# plt.subplot(2, 4, 4)
# plt.imshow(thresh, cmap="gray")
# plt.title("Umbral Otsu invertido")
# plt.axis("off")

# plt.subplot(2, 4,5)
# plt.imshow(mask_circular, cmap="gray")
# plt.title("Mascara circular")
# plt.axis("off")

# plt.subplot(2, 4, 6)
# plt.imshow(D, cmap="jet")
# plt.title("Mapa de distancia")
# plt.axis("off")


# plt.subplot(2, 4, 7)
# plt.imshow(cv2.cvtColor(image_resultado, cv2.COLOR_BGR2RGB))
# plt.title("Colonias detectadas")
# plt.axis("off")

# plt.tight_layout()
# plt.show()

# ...existing code...
# Sustituir la sección de plots por esta versión ampliada con visualizaciones útiles
plt.figure(figsize=(16, 10))

plt.subplot(3, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Imagen original")
plt.axis("off")

plt.subplot(3, 4, 2)
plt.imshow(cv2.cvtColor(imagen_suavizada, cv2.COLOR_BGR2RGB))
plt.title("imagen_suavizada")
plt.axis("off")

plt.subplot(3, 4, 3)
plt.imshow(gray, cmap="gray")
plt.title("Gris")
plt.axis("off")

plt.subplot(3, 4, 4)
plt.imshow(thresh, cmap="gray")
plt.title("Umbral (thresh)")
plt.axis("off")

plt.subplot(3, 4, 5)
plt.imshow(mask_circular, cmap="gray")
plt.title("mask_circular")
plt.axis("off")

plt.subplot(3, 4, 6)
plt.imshow(mask_circular_eroded, cmap="gray")
plt.title("mask_circular_eroded")
plt.axis("off")

plt.subplot(3, 4, 7)
plt.imshow(reflection_mask, cmap="gray")
plt.title("reflection_mask (reflejos)")
plt.axis("off")

plt.subplot(3, 4, 8)
plt.imshow(D, cmap="jet")
plt.title("Mapa de distancia D")
plt.axis("off")

# Mostrar máximos locales sobre el mapa D
plt.subplot(3, 4, 9)
plt.imshow(D, cmap="jet")
if 'coordinates' in globals() and len(coordinates) > 0:
    ys = [c[0] for c in coordinates]
    xs = [c[1] for c in coordinates]
    plt.scatter(xs, ys, c='white', s=20, edgecolors='k')
plt.title("D + máximos locales (coordinates)")
plt.axis("off")

# Overlay de labels/watershed sobre la imagen original (bordes)
plt.subplot(3, 4, 11)
img_labels_overlay = image.copy()
if 'labels' in globals():
    # obtener contorno de la segmentación para visualizar límites
    seg_mask = (labels > 0).astype("uint8") * 255
    edges = cv2.morphologyEx(seg_mask, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    img_labels_overlay[edges > 0] = [255, 0, 0]  # bordes en rojo
plt.imshow(cv2.cvtColor(img_labels_overlay, cv2.COLOR_BGR2RGB))
plt.title("Watershed boundaries")
plt.axis("off")

plt.subplot(3, 4, 12)
plt.imshow(cv2.cvtColor(image_resultado, cv2.COLOR_BGR2RGB))
plt.title("Resultado final")
plt.axis("off")

plt.tight_layout()
plt.show()

