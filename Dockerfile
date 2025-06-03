# Python'un resmi slim imajını temel al
FROM python:3.9.13

# Çalışma dizinini ayarla
WORKDIR /usr/src/app

# Ortam değişkenleri
ENV PYTHONDONTWRITEBYTECODE 1  # .pyc dosyalarının oluşturulmasını engelle
ENV PYTHONUNBUFFERED 1         # Logların anında görünmesini sağla (Docker/Render için iyi)

# Sistem bağımlılıklarını kur
# build-essential: C/C++ kodlarını derlemek için temel araçlar
# pkg-config: Derleme sırasında kütüphane bilgilerini sorgulamak için
# libjpeg-dev, libpng-dev, zlib1g-dev: Pillow gibi görüntü işleme kütüphaneleri için
# tesseract-ocr ve dil paketleri: OCR için
# libtesseract-dev: pytesseract'in Tesseract ile etkileşimi için (bazen C API'leri gerekebilir)
# libopencv-dev: opencv-python'un bazı özelliklerinin derlenmesi veya sistem OpenCV'si ile entegrasyon için
# --no-install-recommends: Önerilen ama zorunlu olmayan paketleri kurmaz, imaj boyutunu küçük tutar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    # Görüntüleme kütüphaneleri için (Pillow, scikit-image vb. kullanabilir)
    libjpeg-dev \
    libpng-dev \
    zlib1g-dev \
    # Tesseract OCR ve geliştirme dosyaları
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-tur \
    libtesseract-dev \
    # OpenCV için geliştirme dosyaları (eğer opencv-python'u kaynaktan derliyorsanız veya özel ihtiyaçlar varsa)
    # Genellikle opencv-python pip ile kurulduğunda kendi binary'lerini getirir,
    # ancak bazı sistem entegrasyonları veya daha karmaşık kullanımlar için libopencv-dev gerekebilir.
    # Eğer opencv-python sorunsuz kuruluyorsa bu satır opsiyonel olabilir veya daha spesifik alt paketleri (libopencv-imgproc-dev vb.) eklenebilir.
    # Şimdilik tam paketi tutalım, sorun çıkarırsa veya imaj boyutu çok büyürse daraltılabilir.
    libopencv-dev \
    # İhtiyaç duyulabilecek diğer kütüphaneler (örneğin, bazı ses/video işleme veya ağ kütüphaneleri için)
    # libffi-dev \ # cffi paketi için
    # libssl-dev \ # cryptography gibi paketler için
    # Temizlik
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pip'i en son sürüme yükselt
RUN pip install --no-cache-dir --upgrade pip

# requirements.txt dosyasını kopyala
# Bu adımı COPY . . 'den önce yapmak, requirements.txt değişmediği sürece
# sonraki pip install adımının Docker build cache'ini kullanmasını sağlar.
COPY requirements.txt .

# Python bağımlılıklarını kur
# --no-cache-dir: pip'in indirme önbelleğini kullanmamasını sağlar, imaj boyutunu küçültür
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarının geri kalanını kopyala
COPY . .

# Flask uygulamasının çalıştığı portu belirt
EXPOSE 10000

# Uygulamayı çalıştırma komutu
# Üretim ortamı için Gunicorn gibi bir WSGI sunucusu kullanılması şiddetle tavsiye edilir.
# Eğer Gunicorn kullanacaksanız, requirements.txt dosyanıza "gunicorn" eklemeyi unutmayın.
# CMD ["python", "app.py"] # Geliştirme sunucusu (üretim için önerilmez)
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
# Yukarıdaki "app:app" -> "dosya_adi:flask_uygulama_nesnesi_adi" şeklinde olmalıdır.
# Eğer dosya adınız app.py ve Flask nesneniz app ise "app:app" doğrudur.