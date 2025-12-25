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
DB_PATH = os.path.join(BASE_DIR, "isg_database.db")

# --- VERİTABANI OLUŞTURMA (SADECE OLUŞTURUR, SİLMEZ) ---
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

# Başlangıçta veritabanını kontrol et
if not os.path.exists(DB_PATH):
    init_db()

# --- MODELLERİ YÜKLE ---
@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_BARET_PATH):
        st.error(f"Model bulunamadı: {MODEL_BARET_PATH}")
        return None, None
    try:
        model_baret = YOLO(MODEL_BARET_PATH)
        # Bot modeli varsa yükle, yoksa None dön
        model_bot = YOLO(MODEL_BOT_PATH) if os.path.exists(MODEL_BOT_PATH) else None
        return model_baret, model_bot
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None, None

model_baret, model_bot = load_models()

# --- RENK TANIMLAMALARI ---
colors = {
    'Hardhat': (0, 255, 0),       # Yeşil
    'Vest': (0, 255, 0),          # Yeşil
    'safety boot': (0, 255, 0),   # Yeşil
    'worker': (255, 191, 0),      # Turuncu
    'NO-Hardhat': (0, 0, 255),    # Kırmızı
    'NO-Vest': (0, 0, 255),       # Kırmızı
    'NO-Safety Boot': (0, 0, 255) # Kırmızı
}

# --- GÖRÜNTÜ İŞLEME SINIFI ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.violation_start_time = None
        self.violation_logged = False
        self.current_violation_label = ""

    def recv(self, frame):
        # 1. Görüntüyü al
        img = frame.to_ndarray(format="bgr24")
        
        # Model yüklenmediyse görüntüyü direkt geri ver (Hata vermesin)
        if model_baret is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # 2. Tespit İşlemi
        results_list = []
        # Baret modeli tahmini
        results_list.append(model_baret(img, conf=0.45, verbose=False))
        # Bot modeli tahmini (varsa)
        if model_bot:
            results_list.append(model_bot(img, conf=0.50, verbose=False))

        violation_detected_in_frame = False
        detected_label = ""

        # 3. Çizim İşlemi (Kritik Kısım)
        for results in results_list:
            for r in results:
                # Her bir kutucuk (box) için dön
                for box in r.boxes:
                    # Koordinatları al (x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Sınıfı ve etiketi al
                    cls = int(box.cls[0])
                    label = r.names[cls] if r.names and cls in r.names else "Unknown"
                    
                    # İhlal mi? (NO- ile başlayanlar veya Maks_Takmiyor)
                    if label.startswith("NO-") or "Maks_Takmiyor" in label:
                        violation_detected_in_frame = True
                        detected_label = label

                    # Rengi seç
                    color = colors.get(label, (255, 0, 255)) # Tanımsızsa Mor yap
                    
                    # KUTUYU ÇİZ
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # ETİKETİ YAZ
                    # Yazının arkasına renkli zemin koy (okunabilirlik için)
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 4. İhlal Kayıt Mantığı (5 Saniye Kuralı)
        if violation_detected_in_frame:
            # Sayaç başlat
            if self.violation_start_time is None:
                self.violation_start_time = time.time()
                self.violation_logged = False
                self.current_violation_label = detected_label
            
            # Geçen süreyi hesapla
            elapsed_time = time.time() - self.violation_start_time
            
            if elapsed_time < 5:
                # Geri sayım
                remaining = 5 - elapsed_time
                cv2.putText(img, f"IHLAL TESPIT EDILIYOR: {int(remaining)+1}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                # Süre doldu, kaydet
                if not self.violation_logged:
                    cv2.putText(img, "VERITABANINA KAYDEDILDI!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    try:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Resmi hazırla
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        stream = io.BytesIO()
                        pil_img.save(stream, format='JPEG')
                        img_byte = stream.getvalue()
                        
                        # Veritabanına yaz
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute("INSERT INTO violations (timestamp, violation_type, image) VALUES (?, ?, ?)",
                                  (timestamp, self.current_violation_label, img_byte))
                        conn.commit()
                        conn.close()
                        
                        self.violation_logged = True
                    except Exception as e:
                        print(f"Kayıt Hatası: {e}")
                else:
                    cv2.putText(img, "KAYITLI IHLAL DEVAM EDIYOR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # İhlal yoksa sayacı sıfırla
            self.violation_start_time = None
            self.violation_logged = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- ARAYÜZ KISMI ---
st.sidebar.title("🔧 Menü")
mod = st.sidebar.radio("Seçiniz:", ["🎥 Saha Kamerası", "👷 Yönetici Paneli"])

if mod == "🎥 Saha Kamerası":
    st.title("🎥 Canlı Denetim Ekranı")
    st.info("Kamera açıldığında tespitler ve kutucuklar otomatik görünecektir.")
    
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    webrtc_streamer(
        key="isg-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif mod == "👷 Yönetici Paneli":
    st.title("👷 Kayıtlı İhlaller")
    if st.button("Yenile"):
        st.rerun()
        
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, timestamp, violation_type, image FROM violations ORDER BY id DESC")
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            st.write("Henüz kayıt yok.")
        else:
            for row in rows:
                r_id, r_ts, r_type, r_img = row
                with st.container(border=True):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        try:
                            st.image(Image.open(io.BytesIO(r_img)), use_container_width=True)
                        except: st.write("Görüntü yok")
                    with col2:
                        st.error(f"{r_type}")
                        st.write(f"Tarih: {r_ts}")