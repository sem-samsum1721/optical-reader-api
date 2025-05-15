FROM python:3.13-slim

# Sistem bağımlılıklarını yükle (Tesseract ve Türkçe dil paketi için)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    tesseract-ocr-tur \
    && rm -rf /var/lib/apt/lists/*

# Tesseract dil dosyaları için ortam değişkeni
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

# Çalışma dizinini ayarla
WORKDIR /app

# Bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Flask uygulamasını başlat
EXPOSE 5000
ENV PORT=5000
CMD ["python", "app.py"]