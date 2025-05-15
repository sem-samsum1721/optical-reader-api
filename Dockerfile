FROM python:3.13-slim

# Sistem bağımlılıklarını yükle
# Tesseract, Türkçe dil paketi VE OpenCV için gerekli olabilecekler
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    tesseract-ocr-tur \
    # OpenCV için gerekli olabilecek sistem kütüphaneleri:
    libgl1-mesa-glx \
    libglib2.0-0 \
    # İsteğe bağlı ama bazen faydalı olabilecek diğerleri:
    # libsm6 \
    # libxext6 \
    # libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Tesseract dil dosyaları için ortam değişkeni
# Debian/Ubuntu'da Tesseract 5.x için genellikle bu yol doğrudur.
# `tesseract --list-langs` komutu ile kontrol edebilirsiniz.
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/

# Çalışma dizinini ayarla
WORKDIR /app

# Önce sadece requirements.txt'yi kopyala ve bağımlılıkları yükle
# Bu sayede requirements.txt değişmediği sürece bu katman cache'lenir
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarının geri kalanını kopyala
COPY . .

# Flask uygulamasını başlat
# Render.com genellikle kendi PORT değişkenini ayarlar, uygulamanızın bunu kullanması gerekir.
# EXPOSE sadece dokümantasyon amaçlıdır ve konteyner dışından hangi porta erişileceğini belirtmez.
EXPOSE 5000
ENV PORT 5000 # Flask uygulamanızın bu değişkenden portu okuduğundan emin olun
CMD ["python", "app.py"]