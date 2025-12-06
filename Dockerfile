# Imagen base ligera
FROM python:3.11-slim AS base

ENV TZ=America/Mexico_City

# Instalar dependencias mínimas del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        gcc \
        tzdata && \
    rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar solo requirements primero (mejor cache)
COPY requirements.txt .

# Instalar dependencias (torch CPU + ultralytics)
RUN pip install --no-cache-dir torch==2.2.0+cpu -f https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Puerto
EXPOSE 8000

# Comando de ejecución (sin reload)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
