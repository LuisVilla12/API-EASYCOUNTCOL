# Usamos la imagen oficial de Python
FROM python:3.11

# Configuramos zona horaria
ENV TZ=America/Mexico_City

# Instalación de dependencias del sistema
RUN apt-get update && \
    apt-get install -y gcc tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Definimos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos los archivos del backend al contenedor
COPY . .

# Instalamos las dependencias de Python con timeout más largo
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Exponemos el puerto que usará la app
EXPOSE 8000

# Comando para arrancar la app con uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]