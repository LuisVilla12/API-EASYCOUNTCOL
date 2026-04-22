FROM python:3.11

ENV TZ=America/Mexico_City

RUN apt-get update && \
    apt-get install -y \
    gcc \
    tzdata \
    libgl1 \
    libglib2.0-0 \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
