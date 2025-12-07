import os
import pandas as pd
import torch
import cv2
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth, SpectralClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np
from ultralytics import YOLO

#Nombre de la imagen y CSV a procesar
BASE = ""
FEATURE_MAP = None

# Hook para extraer P3 (model.11)
def hook_fmap(module, input, output):
    global FEATURE_MAP
    FEATURE_MAP = output 

# Extraer descriptores por detección
def extract_descriptor(box, fmap, img_shape):
    H_img, W_img = img_shape
    _, C, Hf, Wf = fmap.shape   # Ej: [1,128,64,64]

    x0, y0, x1, y1 = map(float, box)

    # Escalar coords imagen → fmap
    sx0 = int(np.floor((x0 / W_img) * Wf))
    sy0 = int(np.floor((y0 / H_img) * Hf))
    sx1 = int(np.ceil((x1 / W_img) * Wf))
    sy1 = int(np.ceil((y1 / H_img) * Hf))

    # Clamping
    sx0 = max(0, min(Wf - 1, sx0))
    sy0 = max(0, min(Hf - 1, sy0))
    sx1 = max(1, min(Wf, sx1))
    sy1 = max(1, min(Hf, sy1))

    # Asegurar tamaño mínimo
    if sx1 <= sx0:
        sx1 = sx0 + 1
        if sx1 > Wf:
            sx0 = Wf - 1
            sx1 = Wf

    if sy1 <= sy0:
        sy1 = sy0 + 1
        if sy1 > Hf:
            sy0 = Hf - 1
            sy1 = Hf

    crop = fmap[0, :, sy0:sy1, sx0:sx1]

    # Si está vacío → fallback a la celda central
    if crop.numel() == 0:
        cx = min(max((sx0 + sx1) // 2, 0), Wf - 1)
        cy = min(max((sy0 + sy1) // 2, 0), Hf - 1)
        print(f"Crop vacío para box {box}, usando celda central ({cx},{cy}) del fmap")
        return fmap[0, :, cy, cx].cpu().numpy()

    # Mean pooling → descriptor 128-d
    desc = crop.mean(dim=(1, 2)).cpu().numpy()
    return desc


def vector_caracteristicas(img_name):
    full_path = os.path.join("ia/resultados/img/", img_name)
    model_path= "ia/YOLO/1_Y12/weights/best.pt"
    global FEATURE_MAP

    # Cargar imagen
    img= cv2.imread(full_path)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen: {img_name}")
    
    H_img, W_img = img.shape[:2]

    # Preprocesado
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)
    
    try:
        model = YOLO(model_path)
        print(" Modelo cargado correctamente.")
    except Exception as e:
        print("Error cargando modelo:", e)

    # Registro de la capa a extraer
    layer = model.model.model[17]
    layer.register_forward_hook(hook_fmap)

    # Obtener device
    try:
        device = model.device
    except Exception:
        try:
            device = next(model.model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor = tensor.to(device)

    # Forward para generar fmap
    with torch.no_grad():
        results = model(img_rgb, verbose=False)
        
    boxes = results[0].boxes.xyxy.cpu().numpy()
        # Ahora sí, results tiene las detecciones
    print("🔍 Detectados:", len(boxes))

    csv_rows = []
    for idx, box in enumerate(boxes):
        desc = extract_descriptor(box, FEATURE_MAP, (H_img, W_img))

        if np.allclose(desc, 0):
            print(f"⚠️ Box {idx} generó descriptores cero: {box}")

        row = [idx, int(box[0]), int(box[1]), int(box[2]), int(box[3])] + desc.tolist()
        csv_rows.append(row)

    # Header
    C = FEATURE_MAP.shape[1] if FEATURE_MAP is not None else 0
    header = ["id", "x0", "y0", "x1", "y1"] + [f"f{i}" for i in range(C)]

    # Asegurar carpeta y usar nombre base (sin extensión) para el CSV
    os.makedirs("ia/resultados/cvs", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_name))[0]
    csv_path = f"ia/resultados/cvs/{base_name}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)
    
    # Regresa la cantidad de cajas detectadas
    return len(boxes)
        
# Función para detectar número óptimo de clusters
def detect_optimal_clusters(X, max_clusters=4):
    """Detecta el número óptimo de clusters usando silhouette score (min 1, max 4)"""
    if X.shape[0] < 2:
        return 1
    
    max_k = min(max_clusters, X.shape[0] - 1)
    if max_k < 2:
        return 1
    
    best_score = -1
    best_k = 1
    
    for k in range(2, max_k + 1):
        try:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels_temp = kmeans_temp.fit_predict(X)
            
            # Validar que silhouette_score tenga valores válidos
            if len(set(labels_temp)) < 2:
                continue
                
            score = silhouette_score(X, labels_temp)
            
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            print(f"⚠️ Error calculando silhouette para k={k}: {e}")
            continue
    
    return best_k


def tratamiento_imagen(name_image):
    # Normalizar ruta y setear BASE a partir del nombre de archivo (sin extensión)
    base_name = os.path.splitext(os.path.basename(name_image))[0]
    global BASE
    BASE = base_name

    # Genera CSV
    count_col=vector_caracteristicas(name_image)

    # Ejecuta clustering y retorna su resultado
    return cloustering(name_image,count_col)

def cloustering(name_image,count_col):
    OUT_DIR = "ia/resultados/clustering_img"
    full_path = os.path.join("ia/resultados/img/", name_image)

    os.makedirs(OUT_DIR, exist_ok=True)

    CSV_PATH = f"ia/resultados/cvs/{BASE}.csv"
    OUTPUT_AGG = os.path.join(OUT_DIR, f"{BASE}.jpg")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"No se encontró el CSV requerido para clustering: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Columnas de features f0...fN
    cols_features = [c for c in df.columns if c.startswith("f")]
    X = df[cols_features].values

    # Normalización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA a 20 dimensiones
    n_components = min(20, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Detectar clusters óptimos (mínimo 1, máximo 4)
    optimal_clusters = detect_optimal_clusters(X_pca, max_clusters=4)
    print(f"Número óptimo de clusters detectado: {optimal_clusters}")

    n_samples = X_pca.shape[0]

    # Si solo hay 1 cluster válido o pocas muestras, asignar etiquetas neutras para métodos que requieren >=2 clusters
    if optimal_clusters == 1 or n_samples < 2:
        # labels_kmeans = np.zeros(n_samples, dtype=int)
        # labels_gmm = np.zeros(n_samples, dtype=int)
        labels_agg = np.zeros(n_samples, dtype=int)
        # labels_spectral = np.zeros(n_samples, dtype=int)
    else:
        # ========== AGGLOMERATIVE ==========
        agg = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
        labels_agg = agg.fit_predict(X_pca)

    # Cargar imagen
    img = cv2.imread(full_path)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen: {full_path}")

    img_agg = img.copy()

    # Paleta BGR para clusters (se reutiliza por si hay >2 clusters)
    PALETTE = [
        (0, 0, 255),    # Rojo
        (0, 255, 0),    # Verde
        (255, 0, 0),    # Azul
        (0, 255, 255),  # Amarillo-cian
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan-yellow
        (128, 0, 128),
        (0, 128, 128),
        (128, 128, 0),
        (0, 0, 128)
    ]

    def make_color_map(labels):
        uniq = sorted(set(labels))
        cmap = {}
        p = 0
        for lab in uniq:
            if lab == -1:
                cmap[lab] = (128, 128, 128)  # gris para ruido
            else:
                cmap[lab] = PALETTE[p % len(PALETTE)]
                p += 1
        return cmap

    cmap_a = make_color_map(labels_agg)

    # Dibujar bounding boxes en la imagen (grosor fino, sin texto)
    for i, row in df.iterrows():
        x0, y0, x1, y1 = int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])
        
        # --- AGGLOMERATIVE ---
        ca = int(labels_agg[i])
        color_a = cmap_a.get(ca, (255,255,255))
        cv2.rectangle(img_agg, (x0, y0), (x1, y1), color_a, 1)
        
    cv2.imwrite(OUTPUT_AGG, img_agg)
    
    return {
        "image_resultado": img_agg,
        "labels": count_col
}
