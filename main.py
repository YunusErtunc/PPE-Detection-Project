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

# --- VERİTABANI KURULUMU ---
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    # İhlal tablosu: Tarih, İhlal Türü, Fotoğraf (blob)
    c.execute('''CREATE TABLE IF NOT EXISTS violations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  violation_type TEXT,
                  image BLOB)''')
    conn.commit()
    return conn

conn = init_db()

# --- MODELLERİ YÜKLE ---
@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_BARET_PATH):
        st.error(f"Model bulunamadı: {MODEL_BARET_PATH}")
        return None, None
    model_baret = YOLO(MODEL_BARET_PATH)
    model_bot = YOLO(MODEL_BOT_PATH) if os.path.exists(MODEL_BOT_PATH) else None
    return model_baret, model_bot

model_baret, model_bot = load_models()

# --- RENKLER ---
colors = {
    'Hardhat': (0, 255, 0), 'Vest': (0, 255, 0), 'safety boot': (0, 255, 0),
    'worker': (255, 191, 0), 'NO-Hardhat': (0, 0, 255), 'NO-Vest': (0, 0, 255),
    'NO-Safety Boot': (0, 0, 255)
}

# --- GÖRÜNTÜ İŞLEME VE MANTIK SINIFI ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Zamanlayıcı değişkenleri
        self.violation_start_time = None  # İhlalin başladığı an
        self.violation_logged = False     # Bu ihlal zaten kaydedildi mi?
        self.current_violation_label = "" # O anki ihlalin adı

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Tespit Yap
        results_list = [model_baret(img, conf=0.45, verbose=False)]
        if model_bot:
            results_list.append(model_bot(img, conf=0.50, verbose=False))

        violation_detected_in_frame = False
        detected_label = ""

        # 2. Çizim ve Kontrol
        for results in results_list:
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = r.names[cls] if r.names and cls in r.names else "Unknown"
                    
                    # Eğer "NO-" ile başlıyorsa ihlaldir
                    if label.startswith("NO-") or "Maks_Takmiyor" in label:
                        violation_detected_in_frame = True
                        detected_label = label

                    color = colors.get(label, (255, 0, 255))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- 3. 5 SANİYE KURALI MANTIĞI ---
        if violation_detected_in_frame:
            # a) Kronometre henüz başlamadıysa başlat
            if self.violation_start_time is None:
                self.violation_start_time = time.time()
                self.violation_logged = False # Yeni bir ihlal başlıyor
                self.current_violation_label = detected_label
            
            # b) Geçen süreyi hesapla
            elapsed_time = time.time() - self.violation_start_time
            remaining_time = 5 - elapsed_time

            if elapsed_time < 5:
                # Ekrana geri sayım yaz (Sarı renk)
                text = f"DIKKAT! Ihlal Kaydediliyor: {int(remaining_time)+1}"
                cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            
            else:
                # c) Süre doldu! Kaydetme zamanı
                if not self.violation_logged:
                    # Ekrana "KAYDEDİLDİ" yaz (Kırmızı renk)
                    cv2.putText(img, "VERITABANINA KAYDEDILDI!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    # Veritabanına Yaz
                    try:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        stream = io.BytesIO()
                        pil_img.save(stream, format='JPEG')
                        img_byte = stream.getvalue()
                        
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        local_conn = sqlite3.connect(DB_PATH)
                        local_c = local_conn.cursor()
                        local_c.execute("INSERT INTO violations (timestamp, violation_type, image) VALUES (?, ?, ?)",
                                        (timestamp, self.current_violation_label, img_byte))
                        local_conn.commit()
                        local_conn.close()
                        
                        print(f"🚨 KAYIT BAŞARILI: {self.current_violation_label}")
                        self.violation_logged = True # Tekrar tekrar kaydetmeyi engelle
                    except Exception as e:
                        print(f"Kayıt Hatası: {e}")
                else:
                    # Zaten kaydedildiyse sadece ekranda uyarısı kalsın
                    cv2.putText(img, "IHLAL DEVAM EDIYOR (KAYITLI)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        else:
            # İhlal yoksa (veya kişi baretini taktıysa) sayacı sıfırla
            if self.violation_start_time is not None:
                print("İhlal sona erdi, sayaç sıfırlandı.")
            self.violation_start_time = None
            self.violation_logged = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- ARAYÜZ SEÇİMİ (SIDEBAR) ---
st.sidebar.title("🔧 Sistem Ayarları")
mod = st.sidebar.radio("Giriş Modu Seçin:", ["🎥 Saha Kamerası", "👷 Şef Paneli (Admin)"])

# --- MOD 1: SAHA KAMERASI ---
if mod == "🎥 Saha Kamerası":
    st.title("🎥 Saha Denetim Modu")
    st.write("Kamera, 5 saniye boyunca kesintisiz ihlal tespit ederse Şef Paneline düşer.")
    
    # --- BURASI DÜZELTİLDİ: STUN SUNUCUSU AYARI ---
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    webrtc_streamer(
        key="isg-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration, # Ayar buraya eklendi
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# --- MOD 2: ŞEF PANELİ ---
elif mod == "👷 Şef Paneli (Admin)":
    st.title("👷 Şef Denetim Paneli")
    st.write("Sahadan gelen ihlal bildirimleri aşağıda listelenir.")
    
    # Yenileme Butonu
    if st.button("🔄 Listeyi Yenile"):
        st.rerun()

    # Verileri Çek
    c = conn.cursor()
    c.execute("SELECT timestamp, violation_type, image FROM violations ORDER BY id DESC")
    rows = c.fetchall()

    if not rows:
        st.info("Henüz bir ihlal kaydı yok. Saha güvenli görünüyor! ✅")
    else:
        for row in rows:
            ts, v_type, img_data = row
            
            # Kart Görünümü
            with st.container():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Resmi Veritabanından Çöz
                    try:
                        image = Image.open(io.BytesIO(img_data))
                        st.image(image, caption="Kanıt Fotoğrafı", use_container_width=True)
                    except:
                        st.error("Resim yüklenemedi")
                
                with col2:
                    st.error(f"🚨 İHLAL TESPİT EDİLDİ: {v_type}")
                    st.write(f"🕒 **Zaman:** {ts}")
                    st.write("---")