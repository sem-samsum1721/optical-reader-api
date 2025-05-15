FROM python:3.13-slim

Sistem bağımlılıklarını yükle (Tesseract için)

RUN apt-get update && apt-get install -y 
tesseract-ocr 
libtesseract-dev 
&& rm -rf /var/lib/apt/lists/*

Çalışma dizinini ayarla

WORKDIR /app

Bağımlılıkları yükle

COPY requirements.txt . RUN pip install --no-cache-dir -r requirements.txt

Proje dosyalarını kopyala

COPY . .

Flask uygulamasını başlat

ENV PORT=5000 CMD ["python", "app.py"]