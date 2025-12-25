import cv2
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
from ultralytics import YOLO
import numpy as np
import os
import sqlite3
import datetime
import time
from PIL import Image
import io

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="İSG Takip Sistemi", page_icon="🏗️", layout="wide")

# --- DOSYA YOLLARI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BARET_PATH = os.path.join(BASE_DIR, "models", "best.pt")
MODEL_BOT_PATH = os.path.join(BASE_DIR, "models", "bot.pt")
DB_PATH = os.path.join(BASE_DIR, "isg_database.db") # Eski, standart isme döndük

# --- VERİTABANI BAĞLANTISI ---
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS violations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  violation_type TEXT,
                  image BLOB)''')
    conn.commit()
    conn.close()

# Dosya yoksa oluştur
if not os.path.exists(DB_PATH):
    init_db()

# --- MODELLERİ YÜKLE ---
@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_BARET_PATH):
        st.error(f"Model Dosyası Bulunamadı: {MODEL_BARET_PATH}")
        return None, None
    try:
        model_baret = YOLO(MODEL_BARET_PATH)
        model_bot = YOLO(MODEL_BOT_PATH) if os.path.exists(MODEL_BOT_PATH) else None
        return model_baret, model_bot
    except Exception as e:
        st.error(f"Model yüklenirken hata: {e}")
        return None, None

model_baret, model_bot = load_models()

# --- RENKLER ---
colors = {
    'Hardhat': (0, 255, 0),       # Yeşil
    'Vest': (0, 255, 0),          # Yeşil
    'safety boot': (0, 255, 0),   # Yeşil
    'worker': (255, 191, 0),      # Turuncu
    'NO-Hardhat': (0, 0, 255),    # Kırmızı
    'NO-Vest': (0, 0, 255),       # Kırmızı
    'NO-Safety Boot': (0, 0, 255) # Kırmızı
}

# --- GÖRÜNTÜ İŞLEME SINIFI (EN ÖNEMLİ KISIM) ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.violation_start_time = None
        self.violation_logged = False
        self.current_violation_label = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Modeller yüklü değilse görüntüyü boş döndürme, aynen ver
        if model_baret is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # 1. Tespit Yap
        results_list = []
        results_list.append(model_baret(img, conf=0.45, verbose=False))
        if model_bot:
            results_list.append(model_bot(img, conf=0.50, verbose=False))

        violation_detected_in_frame = False
        detected_label = ""

        # 2. Kutucukları Çiz (Loop)
        for results in results_list:
            for r in results:
                for box in r.boxes:
                    # Koordinatları al
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    
                    # Etiketi bul
                    label = r.names[cls] if r.names and cls in r.names else "Unknown"
                    
                    # İhlal Kontrolü
                    if label.startswith("NO-") or "Maks_Takmiyor" in label:
                        violation_detected_in_frame = True
                        detected_label = label

                    # Rengi seç
                    color = colors.get(label, (255, 0, 255))
                    
                    # ÇİZİM KOMUTLARI (Bunlar olmazsa kutu görünmez)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Etiket Zemin ve Yazı
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 3. İhlal Mantığı (5 Saniye Kuralı)
        if violation_detected_in_frame:
            if self.violation_start_time is None:
                self.violation_start_time = time.time()
                self.violation_logged = False
                self.current_violation_label = detected_label
            
            elapsed_time = time.time() - self.violation_start_time
            
            if elapsed_time < 5:
                remaining = 5 - elapsed_time
                cv2.putText(img, f"IHLAL: {int(remaining)+1}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                if not self.violation_logged:
                    cv2.putText(img, "KAYDEDILDI!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    try:
                        # Kayıt İşlemi
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        stream = io.BytesIO()
                        pil_img.save(stream, format='JPEG')
                        img_byte = stream.getvalue()
                        
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute("INSERT INTO violations (timestamp, violation_type, image) VALUES (?, ?, ?)",
                                  (timestamp, self.current_violation_label, img_byte))
                        conn.commit()
                        conn.close()
                        self.violation_logged = True
                    except Exception as e:
                        print(f"Db Hatası: {e}")
                else:
                    cv2.putText(img, "KAYITLI!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            self.violation_start_time = None
            self.violation_logged = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- ARAYÜZ (BASİT VE TEMİZ) ---
st.sidebar.title("Menü")
page = st.sidebar.radio("Sayfa:", ["Kamera Modu", "Yönetici Paneli"])

if page == "Kamera Modu":
    st.title("🎥 İSG Denetim Kamerası")
    
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    webrtc_streamer(
        key="isg-cam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif page == "Yönetici Paneli":
    st.title("📋 Kayıtlı İhlaller")
    if st.button("Yenile"):
        st.rerun()
        
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT id, timestamp, violation_type, image FROM violations ORDER BY id DESC")
            rows = c.fetchall()
            conn.close()
            
            if not rows:
                st.info("Kayıt bulunamadı.")
            else:
                for row in rows:
                    r_id, r_ts, r_type, r_img = row
                    with st.container(border=True):
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            try:
                                st.image(Image.open(io.BytesIO(r_img)), use_container_width=True)
                            except: st.error("Görüntü hatası")
                        with c2:
                            st.error(r_type)
                            st.write(r_ts)
        except:
            st.error("Veritabanı okunurken hata oluştu.")