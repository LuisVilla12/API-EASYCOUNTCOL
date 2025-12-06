import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def cargar_imagen(ruta):
    img = cv2.imread(ruta)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta}")
    return img

def preprocesar_imagen(img):
    # Suavizado para reducir ruido
    suavizada = cv2.GaussianBlur(img, (5, 5), 7)
    # Mean Shift para preservar bordes
    shifted = cv2.pyrMeanShiftFiltering(suavizada, 30, 30)
    # Escala de grises
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    return gray, shifted

def aplicar_mascara_y_recortar(img, mask, circle_params):
    # Aplica la máscara circular
    masked = cv2.bitwise_and(img, img, mask=mask)
    if circle_params is not None:
        x, y, r = circle_params
        # Asegura que los valores sean enteros y estén dentro de los límites
        x, y, r = int(x), int(y), int(r)
        x1 = max(x - r, 0)
        y1 = max(y - r, 0)
        x2 = min(x + r, img.shape[1])
        y2 = min(y + r, img.shape[0])
        # Recorta la imagen al bounding box
        cropped = masked[y1:y2, x1:x2]
        return cropped
    else:
        # Si no hay círculo, retorna la imagen completa enmascarada
        return masked

def detectar_circulo(gray, img):
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=gray.shape[0]//2,
        param1=50, param2=30,
        minRadius=gray.shape[0]//4
    )
    mask = np.zeros_like(gray)
    output = img.copy()
    circle_params = None
    if circles is not None:
        circles = np.uint16(np.around(circles))
        (x, y, r) = max(circles[0, :], key=lambda c: c[2])
        cv2.circle(mask, (x, y), r, 255, -1)  # Usa el radio original
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (255, 0, 0), 3)
        print(f"[INFO] Círculo detectado en (x={x}, y={y}), radio={r}")
        circle_params = (x, y, r)
    else:
        print("[WARN] No se detectó círculo, usando toda la imagen.")
        mask[:] = 255
    return mask, output, circle_params

def mostrar_resultados(output, masked):
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Círculo detectado")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
    plt.title("Imagen recortada a la caja Petri")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Colocar la ruta de la imagen")
    args = vars(ap.parse_args())

    img = cargar_imagen(args["image"])
    gray, shifted = preprocesar_imagen(img)
    mask, output, circle_params = detectar_circulo(gray, img)
    cropped = aplicar_mascara_y_recortar(img, mask, circle_params)
    mostrar_resultados(output, cropped)