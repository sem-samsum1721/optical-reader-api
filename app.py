# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pytesseract
import os
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from skimage import measure
from sklearn.cluster import KMeans
from collections import defaultdict
from flask_cors import CORS
import secrets
import re
import traceback

# --- Tesseract Yapılandırması ---
try:
    pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"# KENDİ YOLUNUZU GİRİN
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract versiyonu bulundu: {version}")
except Exception as e:
    print(f"******************** TESSERACT UYARISI *************************")
    print(f"Tesseract yolu ayarlanamadı veya bulunamadı: {e}.")
    print(f"Lütfen Tesseract'ın kurulu olduğundan ve yolun doğru olduğundan emin olun.")
    print(f"*****************************************************************")

# --- Flask Uygulaması ve CORS ---
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app)

# --- Dosya Yükleme Ayarları ---
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Global Veri Saklama ---
scan_process_data = {
    'target_count': 0,
    'saved_results': [],
    'answer_key_set': False,
    'current_puan_per_soru': 1.0
}

# --- Hata Ayıklama Modu ---
DEBUG_MODE = True
DEBUG_FOLDER = os.path.join(os.getcwd(), 'debug_images')
if DEBUG_MODE:
    os.makedirs(DEBUG_FOLDER, exist_ok=True)

def save_debug_image(image, filename_suffix):
    if DEBUG_MODE:
        try:
            debug_filename = f"{secrets.token_hex(4)}_{filename_suffix}.png"
            filepath = os.path.join(DEBUG_FOLDER, debug_filename)
            cv2.imwrite(filepath, image)
            print(f"[DEBUG] Görüntü kaydedildi: {filepath}")
        except Exception as e:
            print(f"[DEBUG] Görüntü kaydetme hatası ({filename_suffix}): {e}")

class OptikOkuyucu:
    def __init__(self):
        self.cevap_anahtari = None
        self.custom_config = r'--oem 3 --psm 6 -l tur+eng'
        self.form_config = {
            'soru_sayisi': 0,
            'secenek_sayisi': 0,
            'isaret_alanlari': None,
            'roi_isim': None,
            'roi_soyisim': None,
            'roi_numara': None,
            'form_analiz_params': {
                'min_alan': 30,
                'max_alan': 1500,
                'min_yuvarlaklik': 0.45,
                'y_tolerance': 20
            },
            'okuma_params': {
                'isaretleme_esigi': 50,
                'coklu_isaret_fark_orani': 0.80
            }
        }

    def cevap_anahtari_ayarla(self, data):
        print("\n--- Cevap Anahtarı Ayarlanıyor ---")
        cevaplar_str = data.get('cevaplar', '')
        try:
            soru_sayisi = int(data.get('soru_sayisi'))
            secenek_sayisi = int(data.get('secenek_sayisi'))
            puan_per_soru_str = data.get('puan_per_soru', str(scan_process_data['current_puan_per_soru']))
            puan_per_soru = float(puan_per_soru_str.replace(',', '.'))

            if puan_per_soru <= 0:
                raise ValueError("Soru başına puan pozitif olmalıdır.")
            if soru_sayisi <= 0 or secenek_sayisi <= 0:
                raise ValueError("Soru ve seçenek sayısı pozitif ve zorunludur.")
            print(f"İstenen: Soru={soru_sayisi}, Seçenek={secenek_sayisi}, Puan/Soru={puan_per_soru}")

        except (ValueError, TypeError, KeyError) as e:
            print(f"Hata: Geçersiz veya eksik giriş formatı: {e}")
            return {'status': 'error', 'message': f'Geçersiz veya eksik giriş: {e}. Soru/seçenek sayısı zorunlu.'}

        cevaplar = [c.strip().upper() for c in cevaplar_str.split(',') if c.strip() or c == '']
        print(f"Alınan Cevaplar: {cevaplar}")

        if self.form_config['soru_sayisi'] != soru_sayisi or \
           self.form_config['secenek_sayisi'] != secenek_sayisi:
            print("[INFO] Soru/seçenek sayısı değişti, form analizi tekrar yapılacak.")
            self.form_config['isaret_alanlari'] = None
        else:
            print("[INFO] Soru/seçenek sayısı aynı, mevcut işaret alanları (varsa) korunacak.")

        self.form_config['soru_sayisi'] = soru_sayisi
        self.form_config['secenek_sayisi'] = secenek_sayisi
        scan_process_data['current_puan_per_soru'] = puan_per_soru

        gecerli_secenekler = [chr(65 + i) for i in range(secenek_sayisi)] + ['']
        for idx, c in enumerate(cevaplar):
            if c not in gecerli_secenekler:
                return {'status': 'error', 'message': f"Geçersiz cevap '{c}' (pozisyon {idx+1})."}

        if len(cevaplar) < soru_sayisi:
            cevaplar.extend([''] * (soru_sayisi - len(cevaplar)))
        elif len(cevaplar) > soru_sayisi:
            cevaplar = cevaplar[:soru_sayisi]

        self.cevap_anahtari = cevaplar
        scan_process_data['answer_key_set'] = True
        print(f"Cevap Anahtarı Ayarlandı (Soru:{soru_sayisi}, Seçenek:{secenek_sayisi}, Puan/Soru:{puan_per_soru}).")
        print("-" * 30)
        return {
            'status': 'success',
            'message': "Cevap anahtarı ayarlandı.",
            'soru_sayisi': soru_sayisi,
            'secenek_sayisi': secenek_sayisi,
            'puan_per_soru': puan_per_soru,
            'cevap_anahtari_goruntule': self.cevap_anahtari
        }

    def _extract_student_info(self, image):
        extracted_info = {'isim': '', 'soyisim': '', 'numara': ''}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape
        rois_config = {
            'isim': self.form_config.get('roi_isim'),
            'soyisim': self.form_config.get('roi_soyisim'),
            'numara': self.form_config.get('roi_numara')
        }
        print("\n--- OCR Denemesi Başlatılıyor (Varsa) ---")
        ocr_performed = False
        for key, roi_coords in rois_config.items():
            if not roi_coords or len(roi_coords) != 4:
                continue
            ocr_performed = True
            y1, y2, x1, x2 = roi_coords
            if not (0 <= y1 < y2 <= h_img and 0 <= x1 < x2 <= w_img):
                print(f"Uyarı: Geçersiz ROI ({key}): ({y1},{y2},{x1},{x2})")
                extracted_info[key] = "ROI_HATA"
                continue
            roi_img = gray[y1:y2, x1:x2]
            roi_blurred = cv2.GaussianBlur(roi_img, (3,3),0)
            roi_thresh = cv2.adaptiveThreshold(roi_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,5)
            save_debug_image(roi_thresh, f"ocr_roi_{key}")
            try:
                text = pytesseract.image_to_string(roi_thresh, config=self.custom_config).strip()
                if key == 'numara':
                    text = re.sub(r'\D', '', text)
                else:
                    text = re.sub(r'[^\w\sığüşöçĞÜŞÖÇİ]', '', text, flags=re.UNICODE)
                    text = ' '.join(text.split()).title()
                extracted_info[key] = text
                print(f"OCR ({key}): Temizlenmiş: '{text}'")
            except pytesseract.TesseractNotFoundError:
                print("Hata: Tesseract executable bulunamadı.")
                return {'isim': 'TESSERACT_HATA'} # Sadece bir anahtar döndürmek yeterli
            except Exception as ocr_err:
                print(f"OCR Hatası ({key}): {ocr_err}")
                extracted_info[key] = "OCR_HATA"
        if not ocr_performed:
            print("[INFO] Tanımlı OCR ROI yok.")
        print("--- OCR Denemesi Tamamlandı ---")
        return extracted_info

    def optik_form_isle(self, image_path, sinav_turu='Test'):
        if not scan_process_data['answer_key_set'] or self.cevap_anahtari is None:
            return {'status': 'error', 'message': "Cevap anahtarı ayarlanmamış."}

        print(f"\n=== Optik Form İşleniyor: {os.path.basename(image_path)} ===")
        current_puan_per_soru = scan_process_data['current_puan_per_soru']
        print(f"Ayarlar: Soru={self.form_config['soru_sayisi']}, Seçenek={self.form_config['secenek_sayisi']}, Puan/Soru={current_puan_per_soru}")

        image = cv2.imread(image_path)
        if image is None:
            return {'status': 'error', 'message': f"Görüntü okunamadı: {image_path}"}

        target_height = 1100
        try:
            h, w = image.shape[:2]
            ratio = target_height / h
            target_width = int(w * ratio)
            image_resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
            save_debug_image(image_resized, "resized")
        except Exception as resize_err:
            return {'status': 'error', 'message': f"Yeniden boyutlandırma hatası: {resize_err}"}

        ocr_results = self._extract_student_info(image_resized.copy())
        if ocr_results.get('isim') == 'TESSERACT_HATA': # Tesseract hatasını kontrol et
            return {'status': 'error', 'message': "Tesseract hatası."}

        try:
            if self.form_config.get('isaret_alanlari') is None:
                print("\n[INFO] Form analizi yapılıyor...")
                self._form_analiz(image_resized.copy())
                if not self.form_config.get('isaret_alanlari'):
                    raise RuntimeError("Form analizi işaret alanı bulamadı.")
                print("[INFO] Form analizi tamamlandı.")
            else:
                print("\n[INFO] Mevcut işaret alanları kullanılıyor.")
        except ValueError as analiz_val_err:
            return {'status': 'error', 'message': f"Form analizi başarısız: {analiz_val_err}."}
        except Exception as analiz_err:
            traceback.print_exc()
            if not self.form_config.get('isaret_alanlari'):
                return {'status': 'error', 'message': f"Form analiz edilemedi: {analiz_err}."}
            print("[UYARI] Yeni analiz başarısız, mevcut alanlar kullanılacak.")

        print("\n[INFO] Optik işaretler okunuyor...")
        try:
            okuma_sonucu = self._optik_oku(image_resized, current_puan_per_soru)
            if not okuma_sonucu or okuma_sonucu.get('status') == 'error':
                error_msg = okuma_sonucu.get('message', 'Bilinmeyen hata') if okuma_sonucu else 'Okuma boş.'
                return {'status': 'error', 'message': error_msg}
            if 'cevaplar' not in okuma_sonucu or 'analiz' not in okuma_sonucu:
                raise ValueError("Okuma sonucu eksik.")
        except Exception as okuma_err:
            traceback.print_exc()
            return {'status': 'error', 'message': f"Optik okuma hatası: {okuma_err}"}

        ogrenci_cevaplari, analiz = okuma_sonucu['cevaplar'], okuma_sonucu['analiz']
        yanlis_sorular_index = [i + 1 for i, og_cev in enumerate(ogrenci_cevaplari)
                                if i < len(self.cevap_anahtari) and og_cev != '' and self.cevap_anahtari[i] != '' and og_cev != self.cevap_anahtari[i]]
        result_payload = {
            'dogru': analiz['dogru'], 'yanlis': analiz['yanlis'], 'bos': analiz['bos'],
            'coklu_isaret': analiz.get('coklu_isaret',0), 'puan': analiz['puan'], 'yuzde': analiz['yuzde'],
            'puan_per_soru': current_puan_per_soru, 'cevap_anahtari': self.cevap_anahtari,
            'ogrenci_cevaplari': ogrenci_cevaplari, 'yanlis_sorular': yanlis_sorular_index,
            'sinav_turu': sinav_turu, 'soru_sayisi': self.form_config['soru_sayisi'],
            'secenek_sayisi': self.form_config['secenek_sayisi'],
            'ocr_isim': ocr_results.get('isim',''), 'ocr_soyisim': ocr_results.get('soyisim',''), 'ocr_numara': ocr_results.get('numara','')
        }
        print(f"\nİşlem Tamamlandı: D={analiz['dogru']} Y={analiz['yanlis']} B={analiz['bos']} PUAN={analiz['puan']:.2f} (Puan/Soru: {current_puan_per_soru})")
        return {'status': 'success', 'message': 'Optik form işlendi.', 'scan_data': result_payload}

    def _form_analiz(self, image):
        print("--- Form Analizi Detayları ---")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5,5),0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19,5)
        save_debug_image(thresh, "analysis_thresh")
        labels = measure.label(thresh, connectivity=2, background=0)
        props = measure.regionprops(labels)
        print(f"1. İlk Etiketleme: {len(props)} bölge.")
        params = self.form_config['form_analiz_params']
        min_alan, max_alan, min_yuvarlaklik, y_tolerance = params['min_alan'], params['max_alan'], params['min_yuvarlaklik'], params['y_tolerance']
        print(f"2. Filtre Parametreleri: Alan=({min_alan}-{max_alan}), Yuvarlaklık>={min_yuvarlaklik:.2f}, Y-Tol={y_tolerance}px")
        bubble_centers = []
        filtered_out = {'small':0,'big':0,'not_round':0}
        image_with_bubbles = image.copy() if DEBUG_MODE else None
        for prop in props:
            area, perimeter = prop.area, prop.perimeter
            if area < min_alan: filtered_out['small']+=1; continue
            if area > max_alan: filtered_out['big']+=1; continue
            if perimeter == 0: continue
            circularity = (4 * np.pi * area) / (perimeter**2)
            if circularity < min_yuvarlaklik: filtered_out['not_round']+=1; continue
            center_y, center_x = map(int, prop.centroid)
            bubble_centers.append((center_y, center_x))
            if DEBUG_MODE: cv2.circle(image_with_bubbles, (center_x, center_y), 10, (0,255,0),1)
        print(f"3. Filtreleme: {len(bubble_centers)} potansiyel baloncuk. Elenenler: {filtered_out}")
        if DEBUG_MODE: save_debug_image(image_with_bubbles, "analysis_detected_bubbles")
        if not bubble_centers: raise ValueError("Filtreleme sonrası baloncuk yok.")
        expected_bubbles = self.form_config['soru_sayisi'] * self.form_config['secenek_sayisi']
        bubble_count = len(bubble_centers)
        print(f"4. Beklenen: {expected_bubbles} (Bulunan: {bubble_count})")
        if abs(bubble_count - expected_bubbles) > expected_bubbles * 0.3:
            print(f"   [UYARI] Bulunan baloncuk sayısı ({bubble_count}) beklenenden ({expected_bubbles}) farklı!")
            if bubble_count < expected_bubbles * 0.4: raise ValueError(f"Çok az baloncuk ({bubble_count}/{expected_bubbles}).")
        n_clusters_kmeans = expected_bubbles
        if n_clusters_kmeans <= 0: raise ValueError("Beklenen baloncuk sayısı pozitif olmalı.")
        if bubble_count < n_clusters_kmeans and bubble_count > 0:
            print(f"   [UYARI] K-Means küme sayısı {bubble_count} olarak ayarlandı (beklenen: {n_clusters_kmeans})")
            n_clusters_kmeans = bubble_count
        print(f"5. K-Means ({n_clusters_kmeans} küme)...")
        kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42, n_init=10)
        try: kmeans.fit(np.array(bubble_centers))
        except ValueError as kmeans_err: raise ValueError(f"K-Means hatası: {kmeans_err}")
        cluster_centers = kmeans.cluster_centers_
        print(f"   K-Means {len(cluster_centers)} merkez buldu.")
        if DEBUG_MODE:
            img_clusters = image_with_bubbles.copy()
            for i, (y,x) in enumerate(cluster_centers):
                cv2.circle(img_clusters,(int(x),int(y)),5,(0,0,255),-1)
                cv2.putText(img_clusters,str(i),(int(x)+5,int(y)+5),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,0,0),1)
            save_debug_image(img_clusters, "analysis_kmeans_centers")
        print(f"6. Sıralama (Y Tol: {y_tolerance}px)...")
        cluster_info = sorted(list(enumerate(cluster_centers)), key=lambda item: item[1][0])
        rows, current_row = [], []
        if cluster_info:
            ref_y = cluster_info[0][1][0]
            for index, (y,x) in cluster_info:
                if abs(y-ref_y) > y_tolerance:
                    current_row.sort(key=lambda item:item[1][1])
                    rows.append(current_row)
                    current_row=[(index,(y,x))]
                    ref_y=y
                else:
                    current_row.append((index,(y,x)))
            if current_row:
                current_row.sort(key=lambda item:item[1][1])
                rows.append(current_row)
        print(f"   {len(rows)} satır/grup bulundu.")
        final_sorted_indices = []
        for r_idx, row_data in enumerate(rows):
            indices = [item[0] for item in row_data]
            print(f"     Satır {r_idx+1} ({len(row_data)} el): {indices}")
            final_sorted_indices.extend(indices)
        print(f"7. Sıralama Sonrası Alan: {len(final_sorted_indices)}")
        if len(final_sorted_indices) != expected_bubbles:
            if abs(len(final_sorted_indices) - expected_bubbles) > expected_bubbles * 0.1:
                raise ValueError(f"Sıralama alanı ({len(final_sorted_indices)}) beklenenden ({expected_bubbles}) çok farklı!")
            print(f"   [UYARI] Sıralama alanı ({len(final_sorted_indices)}) beklenenden ({expected_bubbles}) farklı! Sonuçlar hatalı olabilir.")
        print("8. İşaret Alanları Saklanıyor...")
        clustered_points = defaultdict(list)
        for i,center in enumerate(bubble_centers):
            clustered_points[kmeans.labels_[i]].append(center)
        new_isaret_alanlari = defaultdict(list)
        valid_area_count = 0
        img_final = image.copy() if DEBUG_MODE else None
        for new_idx, original_cluster_index in enumerate(final_sorted_indices):
            if new_idx >= expected_bubbles:
                print(f"   [UYARI] Fazla sıralı indeks ({new_idx}), kırpılıyor.")
                break
            points = clustered_points.get(original_cluster_index,[])
            if points:
                coords = [(int(x_coord),int(y_coord)) for y_coord,x_coord in points]
                new_isaret_alanlari[new_idx]=coords
                valid_area_count+=1
                if DEBUG_MODE:
                    avg_x=int(np.mean([p[0] for p in coords]))
                    avg_y=int(np.mean([p[1] for p in coords]))
                    cv2.putText(img_final,str(new_idx),(avg_x,avg_y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,255),1)
            else:
                new_isaret_alanlari[new_idx]=[]
                print(f"   [UYARI] Sıralı alan {new_idx} (küme {original_cluster_index}) için baloncuk yok.")
        if DEBUG_MODE: save_debug_image(img_final, "analysis_final_order")
        self.form_config['isaret_alanlari'] = new_isaret_alanlari
        print(f"--- Form Analizi Tamamlandı: {valid_area_count}/{expected_bubbles} geçerli alan. ---")

    def _optik_oku(self, image, puan_per_soru):
        if not self.form_config.get('isaret_alanlari'):
            return {'status': 'error', 'message': "İşaret alanları tanımsız."}
        processed = self._goruntu_isle(image)
        save_debug_image(processed, "reading_processed_thresh")
        cev_res = self._cevaplari_bul(processed)
        og_cev, coklu_is = cev_res['cevaplar'], cev_res['coklu_isaret']
        s_sayisi = self.form_config['soru_sayisi']
        if len(og_cev) != s_sayisi:
            print(f"UYARI: Okunan cevap ({len(og_cev)}) soru sayısından ({s_sayisi}) farklı.")
            if len(og_cev) < s_sayisi:
                og_cev.extend(['']*(s_sayisi-len(og_cev)))
            else:
                og_cev = og_cev[:s_sayisi]
        analiz = self._sonuclari_analiz_et(og_cev, coklu_is, puan_per_soru)
        return {'status': 'success', 'cevaplar': og_cev, 'analiz': analiz}

    def _goruntu_isle(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 5)

    def _cevaplari_bul(self, processed_image):
        s_say, sec_say = self.form_config['soru_sayisi'], self.form_config['secenek_sayisi']
        is_alan, params = self.form_config['isaret_alanlari'], self.form_config['okuma_params']
        is_esik, coklu_oran = params['isaretleme_esigi'], params['coklu_isaret_fark_orani']
        cevaplar, coklu_say = ['']*s_say, 0
        if not is_alan:
            return {'cevaplar': cevaplar, 'coklu_isaret': coklu_say}
        h_img, w_img = processed_image.shape
        img_rois = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR) if DEBUG_MODE else None
        for s_idx in range(s_say):
            dolguluklar, max_dolg, is_sec_idx = [], -1, -1
            for sec_idx in range(sec_say):
                alan_idx = s_idx*sec_say+sec_idx
                alan_koor = is_alan.get(alan_idx)
                dolg=0
                if alan_koor:
                    try:
                        pts=np.array(alan_koor,dtype=np.int32)
                        x,y,w,h=cv2.boundingRect(pts)
                        xmin,ymin,xmax,ymax=max(0,x),max(0,y),min(w_img,x+w),min(h_img,y+h)
                        if xmax>xmin and ymax>ymin:
                            roi=processed_image[ymin:ymax,xmin:xmax]
                            if roi.size>0: dolg=np.mean(roi)
                        if DEBUG_MODE:
                            cv2.rectangle(img_rois,(xmin,ymin),(xmax,ymax),(0,int(dolg),255-int(dolg)),1)
                            cv2.putText(img_rois,f"{dolg:.0f}",(xmin,ymin-2),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,255,0),1)
                    except Exception as e:
                        print(f"ROI Hatası S{s_idx+1}-Seç{sec_idx}: {e}")
                dolguluklar.append(dolg)
                if dolg>max_dolg:
                    max_dolg=dolg
                    is_sec_idx=sec_idx
            is_var = (max_dolg>=is_esik and is_sec_idx!=-1)
            is_coklu = False
            if is_var:
                num_sig = sum(1 for d in dolguluklar if d>=is_esik and d>=max_dolg*coklu_oran)
                if num_sig>1:
                    is_coklu=True
                    coklu_say+=1
            if is_coklu: cevaplar[s_idx]=''
            elif is_var: cevaplar[s_idx]=chr(65+is_sec_idx)
            else: cevaplar[s_idx]=''
            if DEBUG_MODE and cevaplar[s_idx]!='':
                alan_idx=s_idx*sec_say+is_sec_idx
                alan_koor=is_alan.get(alan_idx)
                if alan_koor:
                    pts=np.array(alan_koor,dtype=np.int32)
                    x,y,w,h=cv2.boundingRect(pts)
                    cv2.rectangle(img_rois,(x,y),(x+w,y+h),(0,255,0),2)
        if DEBUG_MODE: save_debug_image(img_rois, "reading_bubble_rois")
        return {'cevaplar':cevaplar, 'coklu_isaret':coklu_say}

    def _sonuclari_analiz_et(self, ogrenci_cevaplari, coklu_isaret_sayisi=0, puan_per_soru=1.0):
        dogru,yanlis,bos = 0,0,0
        if not self.cevap_anahtari:
            return {'dogru':0,'yanlis':0,'bos':len(ogrenci_cevaplari),'coklu_isaret':coklu_isaret_sayisi,'puan':0,'yuzde':0}
        for i in range(len(self.cevap_anahtari)):
            og_cev = ogrenci_cevaplari[i] if i<len(ogrenci_cevaplari) else ''
            an_cev = self.cevap_anahtari[i]
            if og_cev=='': bos+=1
            elif an_cev=='': pass
            elif og_cev==an_cev: dogru+=1
            else: yanlis+=1
        puan = float(dogru)*puan_per_soru
        deg_soru = sum(1 for k in self.cevap_anahtari if k!='')
        yuzde = round((dogru/deg_soru)*100,2) if deg_soru>0 else 0.0
        print(f"Analiz: D={dogru},Y={yanlis},B={bos} (Çoklu={coklu_isaret_sayisi}),Puan={puan:.2f} (Puan/Soru:{puan_per_soru}),Yüzde={yuzde:.2f}%")
        return {'dogru':dogru,'yanlis':yanlis,'bos':bos,'coklu_isaret':coklu_isaret_sayisi,'puan':puan,'yuzde':yuzde}

okuyucu = OptikOkuyucu()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/cevap_anahtari', methods=['POST'])
def handle_cevap_anahtari():
    if not request.is_json:
        return jsonify({'status':'error','message':'JSON Gerekli'}),400
    data = request.get_json()
    if not data or 'cevaplar' not in data or 'soru_sayisi' not in data or 'secenek_sayisi' not in data:
        return jsonify({'status':'error','message':"'cevaplar','soru_sayisi','secenek_sayisi' zorunlu."}),400
    result = okuyucu.cevap_anahtari_ayarla(data)
    return jsonify(result), 200 if result.get('status')=='success' else 400

@app.route('/api/set_target_count', methods=['POST'])
def set_target_count():
    if not request.is_json:
        return jsonify({'status':'error','message':'JSON Gerekli'}),400
    data = request.get_json()
    if not data or 'target_count' not in data:
        return jsonify({'status':'error','message':"'target_count' eksik"}),400
    try:
        count = int(data['target_count'])
        if count < 0:
            raise ValueError("Negatif olamaz")
    except:
        return jsonify({'status':'error','message':'Geçersiz target_count'}),400
    scan_process_data['target_count']=count
    scan_process_data['saved_results']=[]
    okuyucu.form_config['isaret_alanlari']=None
    print(f"Hedef {count} ayarlandı.")
    return jsonify({'status':'success','message':f'Hedef {count} ayarlandı.','target_count':count,'saved_count':0}),200

@app.route('/api/optik_oku', methods=['POST'])
def handle_optik_oku():
    if not scan_process_data['answer_key_set']:
        return jsonify({'status':'error','message':'Önce cevap anahtarını ayarla.'}),400
    if 'file' not in request.files:
        return jsonify({'status':'error','message':"'file' bulunamadı."}),400
    file = request.files['file']
    if file.filename=='':
        return jsonify({'status':'error','message':'Dosya seçilmedi.'}),400
    if not allowed_file(file.filename):
        return jsonify({'status':'error','message':"İzin verilmeyen dosya türü."}),400
    filepath=None
    try:
        filename=secure_filename(file.filename)
        unique_filename=f"{secrets.token_hex(8)}_{filename}"
        filepath=os.path.join(UPLOAD_FOLDER,unique_filename)
        file.save(filepath)
        sinav_turu=request.form.get('sinav_turu','Test')
        result=okuyucu.optik_form_isle(filepath,sinav_turu)
        return jsonify(result), 200 if result.get('status')=='success' else 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status':'error','message':f'Sunucu hatası: {str(e)}'}),500
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as re_err: # Değişken adını değiştirdim
                print(f"Uyarı: Dosya silinemedi: {filepath}, Hata: {re_err}")

@app.route('/api/save_result', methods=['POST'])
def save_result():
    if not request.is_json:
        return jsonify({'status':'error','message':'JSON Gerekli'}),400
    data = request.get_json()
    if not data:
        return jsonify({'status':'error','message':'İstek gövdesi boş'}),400
    isim,soyisim,numara = data.get('isim','').strip(),data.get('soyisim','').strip(),data.get('numara','').strip()
    scan_data = data.get('scan_data')
    if not scan_data or not isinstance(scan_data,dict):
        return jsonify({'status':'error','message':"'scan_data' eksik."}),400
    final_record={'isim':isim,'soyisim':soyisim,'numara':numara,'scan_data':scan_data}
    scan_process_data['saved_results'].append(final_record)
    s_count,t_count = len(scan_process_data['saved_results']),scan_process_data['target_count']
    print(f"Sonuç kaydedildi ({s_count}/{t_count if t_count>0 else '∞'}). {isim} {soyisim} ({numara})")
    completed = t_count>0 and s_count>=t_count
    msg = "Sonuç kaydedildi."+(" Hedef tamamlandı." if completed else "")
    return jsonify({'status':'success','message':msg,'completed':completed,'saved_count':s_count,'target_count':t_count}),200

@app.route("/api/saved_results", methods=["GET"])
def get_saved_results():
    results,s_count,t_count = scan_process_data['saved_results'],len(scan_process_data['saved_results']),scan_process_data['target_count']
    total_p,valid_avg = 0.0,0
    for res in results:
        puan = res.get('scan_data',{}).get('puan')
        if isinstance(puan,(int,float)):
            total_p+=puan
            valid_avg+=1
    avg_p = round(total_p/valid_avg,2) if valid_avg>0 else 0.0
    return jsonify({'status':'success','results':results,'saved_count':s_count,'target_count':t_count,'average_puan':avg_p}),200

@app.route("/api/reset_process", methods=["POST"])
def reset_process():
    scan_process_data['target_count']=0
    scan_process_data['saved_results']=[]
    scan_process_data['answer_key_set']=False
    scan_process_data['current_puan_per_soru']=1.0
    okuyucu.cevap_anahtari=None
    okuyucu.form_config['isaret_alanlari']=None
    okuyucu.form_config['soru_sayisi']=0
    okuyucu.form_config['secenek_sayisi']=0
    print("\n--- İşlem Sıfırlandı ---")
    return jsonify({'status':'success','message':'Tarama işlemi ve tüm veriler sıfırlandı.'}),200

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'status':'success','message':'API durumu alındı.',
        'target_count':scan_process_data['target_count'],
        'saved_count':len(scan_process_data['saved_results']),
        'answer_key_set':scan_process_data['answer_key_set'],
        'current_question_count':okuyucu.form_config.get('soru_sayisi') if scan_process_data['answer_key_set'] else None,
        'current_option_count':okuyucu.form_config.get('secenek_sayisi') if scan_process_data['answer_key_set'] else None,
        'current_puan_per_soru':scan_process_data['current_puan_per_soru']
    }),200

@app.route('/')
def index():
    return 'Flask Optik Okuyucu API çalışıyor!'

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=DEBUG_MODE)