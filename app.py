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
from datetime import datetime

# --- Tesseract Configuration ---
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Docker içindeki Tesseract yolu
try:
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract version found: {version}")
except pytesseract.TesseractNotFoundError:
    print("CRITICAL WARNING: Tesseract OCR not found. Ensure it's installed in the Dockerfile.")
except Exception as e:
    print(f"Tesseract version check error: {e}")

# --- Flask Application and CORS ---
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app)

# --- File Upload Settings ---
UPLOAD_FOLDER = '/usr/src/app/Uploads'  # Docker WORKDIR'e göre ayarlandı
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Global Data Storage ---
scan_process_data = {
    'target_count': 0,
    'saved_results': [],
    'answer_key_set': False,
    'current_puan_per_soru': 1.0,
    'current_soru_sayisi_from_key': None,
    'current_secenek_sayisi_from_key': None
}

# --- Debug Mode ---
DEBUG_MODE = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
DEBUG_FOLDER = '/usr/src/app/debug_images'  # Docker WORKDIR'e göre ayarlandı
if DEBUG_MODE:
    os.makedirs(DEBUG_FOLDER, exist_ok=True)
    print("WARNING: Flask DEBUG mode ACTIVE!")
else:
    print("Flask DEBUG mode OFF.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_debug_image(image, filename_suffix):
    if DEBUG_MODE:
        try:
            safe_suffix = re.sub(r'[^\w\s_.)( -]', '', filename_suffix).replace(" ", "_")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            debug_filename = f"{timestamp}_{safe_suffix}.png"
            filepath = os.path.join(DEBUG_FOLDER, debug_filename)
            cv2.imwrite(filepath, image)
            print(f"[DEBUG] Image saved: {filepath}")
        except Exception as e:
            print(f"[DEBUG] Image save error ({filename_suffix}): {e}")
            traceback.print_exc()

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
                'isaretleme_esigi': 65,
                'coklu_isaret_fark_orani': 0.75,
                'roi_yari_boyut': 8,
                'ocr_timeout': 10  # Timeout'u 20'den 10'a düşürdüm, performans için
            }
        }
        print("[OptikOkuyucu] Initialized.")

    def cevap_anahtari_ayarla(self, data):
        print("\n--- [API] Answer Key Setup Request Received ---")
        print(f"[DEBUG] Received JSON: {json.dumps(data, indent=2, ensure_ascii=False)}")

        if not isinstance(data, dict):
            print("[API ERROR] Invalid data format. Expected JSON.")
            scan_process_data['answer_key_set'] = False
            return {'status': 'error', 'message': 'Invalid data format. JSON expected.'}

        required_fields = ['answers', 'soru_sayisi', 'option_count']
        optional_fields = {'puan_per_soru': '1.0', 'roi_isim': None, 'roi_soyisim': None, 'roi_numara': None}
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            error_message = f"Missing required fields: {', '.join(missing_fields)}. Expected: 'answers', 'soru_sayisi', 'option_count'."
            print(f"[API ERROR] {error_message}")
            scan_process_data['answer_key_set'] = False
            return {'status': 'error', 'message': error_message}

        try:
            cevaplar_str = data['answers']
            soru_sayisi = int(data['soru_sayisi'])
            secenek_sayisi = int(data['option_count'])
            puan_per_soru = float(data.get('puan_per_soru', optional_fields['puan_per_soru']).replace(',', '.'))

            self.form_config['roi_isim'] = data.get('roi_isim', optional_fields['roi_isim'])
            self.form_config['roi_soyisim'] = data.get('roi_soyisim', optional_fields['roi_soyisim'])
            self.form_config['roi_numara'] = data.get('roi_numara', optional_fields['roi_numara'])
            print(f"[INFO] Received ROIs: isim={self.form_config['roi_isim']}, soyisim={self.form_config['roi_soyisim']}, numara={self.form_config['roi_numara']}")

            if soru_sayisi <= 0:
                raise ValueError("Question count must be a positive integer.")
            if secenek_sayisi <= 0:
                raise ValueError("Option count must be a positive integer.")
            if puan_per_soru <= 0:
                raise ValueError("Score per question must be a positive value.")

            if isinstance(cevaplar_str, list):
                gelen_cevap_listesi = cevaplar_str
            else:
                gelen_cevap_listesi = cevaplar_str.split(',')

            if len(gelen_cevap_listesi) != soru_sayisi:
                raise ValueError(f"Answer count ({len(gelen_cevap_listesi)}) does not match question count ({soru_sayisi}).")

            cevaplar = [c.strip().upper() for c in gelen_cevap_listesi]
            gecerli_secenekler_harf = [chr(65 + i) for i in range(secenek_sayisi)]

            for idx, c in enumerate(cevaplar):
                if c != '' and c not in gecerli_secenekler_harf:
                    raise ValueError(f"Invalid answer '{c}' (question {idx+1}). Allowed: {', '.join(gecerli_secenekler_harf)} or empty.")

            if (self.form_config.get('soru_sayisi', 0) != soru_sayisi or
                    self.form_config.get('secenek_sayisi', 0) != secenek_sayisi):
                self.form_config['isaret_alanlari'] = None

            self.form_config['soru_sayisi'] = soru_sayisi
            self.form_config['secenek_sayisi'] = secenek_sayisi
            self.cevap_anahtari = cevaplar

            scan_process_data['current_puan_per_soru'] = puan_per_soru
            scan_process_data['current_soru_sayisi_from_key'] = soru_sayisi
            scan_process_data['current_secenek_sayisi_from_key'] = secenek_sayisi
            scan_process_data['answer_key_set'] = True

            print(f"Answer Key Set Successfully: {self.cevap_anahtari}")
            return {
                'status': 'success',
                'message': 'Answer key set successfully.',
                'current_question_count': soru_sayisi,
                'current_option_count': secenek_sayisi,
                'current_score_per_question': puan_per_soru
            }

        except ValueError as ve:
            print(f"[API ERROR] ValueError: {ve}")
            traceback.print_exc()
            scan_process_data['answer_key_set'] = False
            return {'status': 'error', 'message': str(ve)}
        except Exception as e:
            print(f"[API ERROR] General Error: {e}")
            traceback.print_exc()
            scan_process_data['answer_key_set'] = False
            return {'status': 'error', 'message': f"Unexpected error: {str(e)}"}

    def _extract_student_info(self, image):
        print("\n--- Student Info OCR Attempt ---")
        extracted_info = {'isim': '', 'soyisim': '', 'numara': ''}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape

        rois_config = {
            'isim': self.form_config.get('roi_isim'),
            'soyisim': self.form_config.get('roi_soyisim'),
            'numara': self.form_config.get('roi_numara')
        }
        ocr_timeout = self.form_config['okuma_params'].get('ocr_timeout', 10)

        for key, roi_coords in rois_config.items():
            if not roi_coords or not (isinstance(roi_coords, list) and len(roi_coords) == 4):
                print(f"[INFO] ROI for '{key}' not defined or invalid. Skipping.")
                continue

            y1, y2, x1, x2 = roi_coords
            y1, y2 = max(0, y1), min(h_img, y2)
            x1, x2 = max(0, x1), min(w_img, x2)

            if not (y1 < y2 and x1 < x2):
                print(f"WARNING: Invalid or out-of-bounds ROI coordinates ({key}): Original ({roi_coords}), Adjusted ({y1},{y2},{x1},{x2}). Skipping.")
                continue

            roi_img = gray[y1:y2, x1:x2]
            if roi_img.size == 0:
                print(f"WARNING: ROI ({key}) is empty or invalid. Skipping.")
                continue

            roi_blurred = cv2.GaussianBlur(roi_img, (3, 3), 0)
            roi_thresh = cv2.adaptiveThreshold(roi_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 11, 7)
            save_debug_image(roi_thresh, f"ocr_roi_thresh_{key}")

            try:
                psm_config = '--psm 7' if key == 'numara' else '--psm 6'
                text = pytesseract.image_to_string(
                    roi_thresh,
                    config=f'{self.custom_config} {psm_config}',
                    timeout=ocr_timeout
                ).strip()

                if key == 'numara':
                    text = re.sub(r'\D', '', text)
                else:
                    text = re.sub(r'[^\w\sığüşöçĞÜŞÖÇİ]', '', text, flags=re.UNICODE)
                    text = ' '.join(word.capitalize() for word in text.split())

                extracted_info[key] = text
                print(f"OCR Result ({key}): '{text}' (Timeout: {ocr_timeout}s)")
            except RuntimeError as e:
                if 'Tesseract process timeout' in str(e) or 'Future not completed' in str(e):
                    print(f"OCR Error ({key}): Tesseract timed out ({ocr_timeout}s). {e}")
                    extracted_info[key] = "OCR_TIMEOUT"
                else:
                    print(f"OCR Runtime Error ({key}): {e}")
                    extracted_info[key] = "OCR_RUNTIME_ERROR"
                    traceback.print_exc()
            except Exception as e:
                print(f"OCR General Error ({key}): {e}")
                extracted_info[key] = "OCR_GENERAL_ERROR"
                traceback.print_exc()

        print("--- Student Info OCR Completed ---")
        return extracted_info

    def optik_form_isle(self, image_path, sinav_turu='Test'):
        print(f"\n=== Optical Form Processing Started: {os.path.basename(image_path)} ===")

        image_original = cv2.imread(image_path)
        if image_original is None:
            return {'status': 'error', 'message': 'Image file could not be read or is corrupted.'}

        target_height = 1100
        h, w = image_original.shape[:2]
        ratio = target_height / h
        target_width = int(w * ratio)
        image_resized = cv2.resize(image_original, (target_width, target_height),
                                  interpolation=cv2.INTER_AREA)
        save_debug_image(image_resized, "resized_form_image")

        try:
            ocr_results = self._extract_student_info(image_resized.copy())
        except Exception as e_ocr:
            print(f"[CRITICAL OCR ERROR] Unexpected error in _extract_student_info: {e_ocr}")
            traceback.print_exc()
            ocr_results = {'isim': 'OCR_SYSTEM_ERROR', 'soyisim': 'OCR_SYSTEM_ERROR', 'numara': 'OCR_SYSTEM_ERROR'}

        if not scan_process_data['answer_key_set']:
            message = ("Answer key (including question/option count) not set. "
                       "Only student info (if ROI defined) extracted via OCR.")
            print(f"[WARNING] {message}")

            num_questions_placeholder = self.form_config.get('soru_sayisi', 0)
            num_options_placeholder = self.form_config.get('secenek_sayisi', 0)

            result_payload = {
                'dogru': 0, 'yanlis': 0, 'bos': num_questions_placeholder, 'coklu_isaret': 0,
                'puan': 0.0, 'yuzde': 0.0,
                'puan_per_soru': scan_process_data.get('current_puan_per_soru', 1.0),
                'cevap_anahtari': [],
                'ogrenci_cevaplari': [''] * num_questions_placeholder,
                'yanlis_sorular': [],
                'sinav_turu': sinav_turu,
                'soru_sayisi': num_questions_placeholder,
                'secenek_sayisi': num_options_placeholder,
                'ocr_isim': ocr_results.get('isim', ''),
                'ocr_soyisim': ocr_results.get('soyisim', ''),
                'ocr_numara': ocr_results.get('numara', '')
            }
            return {'status': 'partial_success', 'message': message, 'scan_data': result_payload}

        if self.form_config.get('isaret_alanlari') is None:
            try:
                self._form_analiz(image_resized.copy())
            except Exception as e:
                error_msg = f"Form analysis error: {str(e)}"
                print(f"[ERROR] {error_msg}\n{traceback.format_exc()}")
                return {
                    'status': 'error',
                    'message': error_msg,
                    'scan_data': {
                        'ocr_isim': ocr_results.get('isim', ''),
                        'ocr_soyisim': ocr_results.get('soyisim', ''),
                        'ocr_numara': ocr_results.get('numara', ''),
                        'dogru': 0, 'yanlis': 0, 'bos': self.form_config.get('soru_sayisi', 0),
                        'coklu_isaret': 0, 'puan': 0.0, 'yuzde': 0.0,
                        'puan_per_soru': scan_process_data.get('current_puan_per_soru', 1.0),
                        'cevap_anahtari': self.cevap_anahtari if self.cevap_anahtari is not None else [],
                        'ogrenci_cevaplari': [''] * self.form_config.get('soru_sayisi', 0),
                        'yanlis_sorular': [],
                        'sinav_turu': sinav_turu,
                        'soru_sayisi': self.form_config.get('soru_sayisi', 0),
                        'secenek_sayisi': self.form_config.get('secenek_sayisi', 0)
                    }
                }

        try:
            okuma_sonucu = self._optik_oku(image_resized.copy(), scan_process_data['current_puan_per_soru'])
            if okuma_sonucu.get('status') == 'error':
                okuma_sonucu['scan_data'] = {
                    'ocr_isim': ocr_results.get('isim', ''),
                    'ocr_soyisim': ocr_results.get('soyisim', ''),
                    'ocr_numara': ocr_results.get('numara', '')
                }
                return okuma_sonucu
        except Exception as e:
            error_msg = f"Optical reading error: {str(e)}"
            print(f"[ERROR] {error_msg}\n{traceback.format_exc()}")
            return {
                'status': 'error',
                'message': error_msg,
                'scan_data': {
                    'ocr_isim': ocr_results.get('isim', ''),
                    'ocr_soyisim': ocr_results.get('soyisim', ''),
                    'ocr_numara': ocr_results.get('numara', '')
                }
            }

        ogrenci_cevaplari, analiz_detaylari = okuma_sonucu['cevaplar'], okuma_sonucu['analiz']

        yanlis_sorular_idx = []
        if self.cevap_anahtari:
            yanlis_sorular_idx = [i for i, (og, an) in enumerate(zip(ogrenci_cevaplari, self.cevap_anahtari))
                                  if og and an and og != an]

        result_payload = {
            'dogru': analiz_detaylari.get('dogru', 0),
            'yanlis': analiz_detaylari.get('yanlis', 0),
            'bos': analiz_detaylari.get('bos', 0),
            'coklu_isaret': analiz_detaylari.get('coklu_isaret', 0),
            'puan': analiz_detaylari.get('puan', 0.0),
            'yuzde': analiz_detaylari.get('yuzde', 0.0),
            'puan_per_soru': scan_process_data['current_puan_per_soru'],
            'cevap_anahtari': self.cevap_anahtari,
            'ogrenci_cevaplari': ogrenci_cevaplari,
            'yanlis_sorular': yanlis_sorular_idx,
            'sinav_turu': sinav_turu,
            'soru_sayisi': self.form_config['soru_sayisi'],
            'secenek_sayisi': self.form_config['secenek_sayisi'],
            'ocr_isim': ocr_results.get('isim', ''),
            'ocr_soyisim': ocr_results.get('soyisim', ''),
            'ocr_numara': ocr_results.get('numara', '')
        }

        print(f"Processing Completed: Correct={result_payload['dogru']}, Incorrect={result_payload['yanlis']}")
        return {'status': 'success', 'message': 'Optical form processed successfully.', 'scan_data': result_payload}

    def _form_analiz(self, image):
        print("\n--- Form Analysis Started (Bubble Detection) ---")

        if self.form_config['soru_sayisi'] <= 0 or self.form_config['secenek_sayisi'] <= 0:
            raise ValueError("Question and option counts must be positive for form analysis.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 19, 7)
        save_debug_image(thresh, "form_analysis_adaptive_threshold")

        labels = measure.label(thresh, connectivity=2, background=0)
        props = measure.regionprops(labels)
        print(f"Total regions found: {len(props)}")

        params = self.form_config['form_analiz_params']
        potential_bubble_centers = []

        for prop in props:
            area = prop.area
            if area < params['min_alan'] or area > params['max_alan']:
                continue
            perimeter = prop.perimeter
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity < params['min_yuvarlaklik']:
                continue
            center_y, center_x = int(prop.centroid[0]), int(prop.centroid[1])
            potential_bubble_centers.append((center_y, center_x))

        print(f"Potential bubbles after filtering: {len(potential_bubble_centers)}")

        expected_total_bubbles = self.form_config['soru_sayisi'] * self.form_config['secenek_sayisi']

        if not potential_bubble_centers:
            raise ValueError(f"Form analysis: No potential marking areas (bubbles) found. Expected: {expected_total_bubbles}")

        num_clusters = min(len(potential_bubble_centers), expected_total_bubbles)
        if num_clusters <= 0:
            raise ValueError(f"Invalid cluster count ({num_clusters}). Are question/option counts correct? Potential bubbles: {len(potential_bubble_centers)}")

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        kmeans.fit(np.array(potential_bubble_centers))

        grouped_bubble_centers_coords = kmeans.cluster_centers_
        sorted_grouped_centers = sorted(enumerate(grouped_bubble_centers_coords),
                                        key=lambda item: (item[1][0], item[1][1]))
        sorted_final_bubble_coords = [coords for _, coords in sorted_grouped_centers]

        if len(sorted_final_bubble_coords) < expected_total_bubbles:
            print(f"WARNING: Expected bubble count ({expected_total_bubbles}) exceeds found ({len(sorted_final_bubble_coords)}). Some questions/options may not be read.")
        elif len(sorted_final_bubble_coords) > expected_total_bubbles:
            print(f"WARNING: Found bubble count ({len(sorted_final_bubble_coords)}) exceeds expected ({expected_total_bubbles}). Excess will be ignored.")

        final_marker_rois = {}
        for i in range(min(len(sorted_final_bubble_coords), expected_total_bubbles)):
            final_marker_rois[i] = (int(sorted_final_bubble_coords[i][0]),
                                    int(sorted_final_bubble_coords[i][1]))

        self.form_config['isaret_alanlari'] = final_marker_rois
        print(f"{len(self.form_config['isaret_alanlari'])} marking areas defined.")
        print("--- Form Analysis Completed ---")

    def _optik_oku(self, image, puan_per_soru):
        print("\n--- Optical Reading Started (Answer Detection) ---")

        if not self.form_config.get('isaret_alanlari'):
            return {'status': 'error', 'message': 'Marking areas (bubbles) not defined. Form analysis required.'}

        processed_thresh_image = self._goruntu_isle(image)
        save_debug_image(processed_thresh_image, "reading_processed_threshold_image")

        cevap_bulma_sonucu = self._cevaplari_bul(processed_thresh_image)
        analiz_sonuclari = self._sonuclari_analiz_et(cevap_bulma_sonucu['cevaplar'],
                                                     cevap_bulma_sonucu['coklu_isaret'],
                                                     puan_per_soru)

        print("--- Optical Reading Completed ---")
        return {'status': 'success', 'cevaplar': cevap_bulma_sonucu['cevaplar'],
                'analiz': analiz_sonuclari}

    def _goruntu_isle(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 19, 7)
        return thresh

    def _cevaplari_bul(self, processed_image):
        soru_sayisi = self.form_config['soru_sayisi']
        secenek_sayisi = self.form_config['secenek_sayisi']
        isaret_alan_merkezleri = self.form_config['isaret_alanlari']
        okuma_params = self.form_config['okuma_params']
        h_img, w_img = processed_image.shape

        ogrenci_cevaplari = [''] * soru_sayisi
        coklu_isaret_sayisi = 0

        if not isaret_alan_merkezleri:
            print("WARNING: Marking area centers not found, cannot read answers.")
            return {'cevaplar': ogrenci_cevaplari, 'coklu_isaret': coklu_isaret_sayisi}

        for s_idx in range(soru_sayisi):
            dolguluk_degerleri = []

            for o_idx in range(secenek_sayisi):
                global_baloncuk_idx = s_idx * secenek_sayisi + o_idx
                merkez_koordinati = isaret_alan_merkezleri.get(global_baloncuk_idx)
                dolguluk = 0.0

                if merkez_koordinati:
                    merkez_y, merkez_x = merkez_koordinati
                    roi_yari_boyut = okuma_params['roi_yari_boyut']

                    y1 = max(0, merkez_y - roi_yari_boyut)
                    x1 = max(0, merkez_x - roi_yari_boyut)
                    y2 = min(h_img, merkez_y + roi_yari_boyut)
                    x2 = min(w_img, merkez_x + roi_yari_boyut)

                    if y2 > y1 and x2 > x1:
                        roi_baloncuk = processed_image[y1:y2, x1:x2]
                        dolguluk = np.mean(roi_baloncuk) if roi_baloncuk.size > 0 else 0.0
                else:
                    print(f"WARNING: Center not found for Question {s_idx+1}, Option {chr(65+o_idx)} ({global_baloncuk_idx}).")

                dolguluk_degerleri.append(dolguluk)

            if not dolguluk_degerleri:
                continue

            max_dolguluk = max(dolguluk_degerleri) if dolguluk_degerleri else 0.0
            isaretli_secenek_idx = dolguluk_degerleri.index(max_dolguluk) if max_dolguluk > 0 else -1

            if max_dolguluk >= okuma_params['isaretleme_esigi'] and isaretli_secenek_idx != -1:
                dikkate_deger_isaretler = [
                    d for d_idx, d in enumerate(dolguluk_degerleri)
                    if d >= okuma_params['isaretleme_esigi'] and
                       d >= max_dolguluk * okuma_params['coklu_isaret_fark_orani']
                ]

                if len(dikkate_deger_isaretler) > 1:
                    coklu_isaret_sayisi += 1
                else:
                    ogrenci_cevaplari[s_idx] = chr(65 + isaretli_secenek_idx)

        print(f"Found Student Answers: {ogrenci_cevaplari}")
        print(f"Multiple Mark Count: {coklu_isaret_sayisi}")
        return {'cevaplar': ogrenci_cevaplari, 'coklu_isaret': coklu_isaret_sayisi}

    def _sonuclari_analiz_et(self, ogrenci_cevaplar, coklu_isaret_sayisi, puan_per_soru):
        if self.cevap_anahtari is None:
            print("WARNING: Answer key is None in _sonuclari_analiz_et. Scoring not possible.")
            bos_ogrenci = sum(1 for c in ogrenci_cevaplar if not c)
            return {
                'dogru': 0, 'yanlis': 0, 'bos': bos_ogrenci,
                'coklu_isaret': coklu_isaret_sayisi, 'puan': 0.0, 'yuzde': 0.0
            }

        dogru_sayisi = 0
        yanlis_sayisi = 0
        bos_sayisi = 0

        for i in range(len(self.cevap_anahtari)):
            og_cvp = ogrenci_cevaplar[i] if i < len(ogrenci_cevaplar) else ''
            anahtar_cvp = self.cevap_anahtari[i]

            if og_cvp == '':
                bos_sayisi += 1
            elif anahtar_cvp == '':
                bos_sayisi += 1
            elif og_cvp == anahtar_cvp:
                dogru_sayisi += 1
            else:
                yanlis_sayisi += 1

        degerlendirilecek_soru_sayisi = sum(1 for c_anahtar in self.cevap_anahtari if c_anahtar)
        puan = dogru_sayisi * puan_per_soru
        basari_yuzdesi = (dogru_sayisi / degerlendirilecek_soru_sayisi * 100) if degerlendirilecek_soru_sayisi > 0 else 0.0

        return {
            'dogru': dogru_sayisi, 'yanlis': yanlis_sayisi, 'bos': bos_sayisi,
            'coklu_isaret': coklu_isaret_sayisi, 'puan': puan, 'yuzde': basari_yuzdesi
        }

# --- Flask Application Routes ---
okuyucu = OptikOkuyucu()

@app.route('/api/cevap_anahtari', methods=['POST'])
def handle_cevap_anahtari():
    if not request.is_json:
        print("[API ERROR] Request is not JSON.")
        return jsonify({'status': 'error', 'message': 'JSON format required.'}), 400

    data = request.get_json()
    print(f"[DEBUG] Received JSON payload: {json.dumps(data, indent=2, ensure_ascii=False)}")
    result = okuyucu.cevap_anahtari_ayarla(data)
    status_code = 200 if result.get('status') == 'success' else 400
    return jsonify(result), status_code

@app.route('/api/set_hedef_sayi', methods=['POST'])
def set_hedef_sayi():
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'JSON format required.'}), 400
    data = request.get_json()
    hedef_sayi = data.get('hedef_sayi')
    if hedef_sayi is None:
        return jsonify({'status': 'error', 'message': "'hedef_sayi' field required."}), 400
    try:
        count = int(hedef_sayi)
        if count < 0:
            raise ValueError("Negative number not allowed.")
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': "Invalid 'hedef_sayi' value."}), 400
    scan_process_data['target_count'] = count
    scan_process_data['saved_results'] = []
    return jsonify({'status': 'success', 'message': f'Target count set to {count}.', 'hedef_sayi': count}), 200

@app.route('/api/tara_optik', methods=['POST'])
def handle_tara_optik():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': "No file provided."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected.'}), 400
    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid file type.'}), 415

    filename = secure_filename(file.filename)
    unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)

    try:
        file.save(filepath)
        result = okuyucu.optik_form_isle(filepath, request.form.get('sinav_turu', 'Test'))
        status_code = 200 if result.get('status') in ['success', 'partial_success'] else 400
        return jsonify(result), status_code
    except Exception as e:
        print(f"[CRITICAL ERROR] /api/tara_optik endpoint: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f"Server error: {str(e)}"}), 500
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e_remove:
                print(f"WARNING: Failed to delete uploaded file: {filepath}, Error: {e_remove}")

@app.route('/api/kaydet_sonuclari', methods=['POST'])
def handle_kaydet_sonuclari():
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'JSON format required.'}), 400
    data = request.get_json()
    required_fields = ['ad', 'soyad', 'numara', 'scan_data']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'status': 'error', 'message': f"Missing fields: {', '.join(missing_fields)}"}), 400
    if not isinstance(data.get('scan_data'), dict):
        return jsonify({'status': 'error', 'message': "Invalid 'scan_data' format."}), 400

    record = {
        'ad': data['ad'].strip().capitalize(),
        'soyad': data['soyad'].strip().upper(),
        'numara': data['numara'].strip(),
        'scan_data': data['scan_data'],
        'kayit_zamani': datetime.now().isoformat()
    }
    scan_process_data['saved_results'].append(record)
    kaydedilmis_sayi = len(scan_process_data['saved_results'])
    hedef_sayi = scan_process_data['target_count']
    hedef_tamamlandi = (hedef_sayi > 0 and kaydedilmis_sayi >= hedef_sayi)
    return jsonify({
        'status': 'success',
        'message': 'Result saved.',
        'completed': hedef_tamamlandi,
        'kaydedilmis_sayi': kaydedilmis_sayi
    }), 200

@app.route('/api/kaydedilmis_sonuc', methods=['GET'])
def get_kaydedilmis_sonclar():
    results = scan_process_data['saved_results']
    return jsonify({
        'status': 'success',
        'results': results,
        'kaydedilmis_sayi': len(results),
        'hedef_sayi': scan_process_data['target_count']
    }), 200

@app.route('/api/sifirla_islem', methods=['POST'])
def sifirla_islemi():
    global okuyucu
    scan_process_data.update({
        'target_count': 0,
        'saved_results': [],
        'answer_key_set': False,
        'current_puan_per_soru': 1.0,
        'current_soru_sayisi_from_key': None,
        'current_secenek_sayisi_from_key': None
    })
    okuyucu = OptikOkuyucu()
    print("\n--- [API] All Process Data Reset ---")
    return jsonify({'status': 'success', 'message': 'All process data and reader state reset.'}), 200

@app.route('/api/status', methods=['GET'])
def get_api_status():
    return jsonify({
        'status': 'success',
        'message': 'API status and current configuration.',
        'target_count': scan_process_data['target_count'],
        'saved_count': len(scan_process_data['saved_results']),
        'answer_key_set': scan_process_data['answer_key_set'],
        'current_question_count_from_key': scan_process_data['current_soru_sayisi_from_key'],
        'current_option_count_from_key': scan_process_data['current_secenek_sayisi_from_key'],
        'current_score_per_question': scan_process_data['current_puan_per_soru'],
        'form_config_question_count': okuyucu.form_config['soru_sayisi'],
        'form_config_option_count': okuyucu.form_config['secenek_sayisi'],
        'ocr_timeout_setting': okuyucu.form_config['okuma_params'].get('ocr_timeout')
    }), 200

@app.route('/')
def index():
    return '<h1>Optical Reader API Running</h1><p>Visit /api/status to check API status.</p>'

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)  # Docker ile uyumlu port