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
    c.execute('''CREATE TABLE IF NOT EXISTS violations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  violation_type TEXT,
                  image BLOB)''')
    conn.commit()
    conn.close()

if not os.path.exists(DB_PATH):
    init_db()

# --- MODELLERİ YÜKLE ---
@st.cache_resource
def load_models():
    # Model dosyası var mı kontrol et
    if not os.path.exists(MODEL_BARET_PATH):
        st.error(f"Model bulunamadı: {MODEL_BARET_PATH}")
        return None, None
    
    # Modelleri yükle
    try:
        model_baret = YOLO(MODEL_BARET_PATH)
        # Bot modeli opsiyonel, varsa yükle
        model_bot = YOLO(MODEL_BOT_PATH) if os.path.exists(MODEL_BOT_PATH) else None
        return model_baret, model_bot
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None, None

model_baret, model_bot = load_models()

# --- RENKLER (GÖRSELLEŞTİRME İÇİN) ---
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
        img = frame.to_ndarray(format="bgr24")
        
        # Eğer modeller yüklenemediyse görüntüyü olduğu gibi döndür
        if model_baret is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # 1. Baret Modelini Çalıştır
        results_list = [model_baret(img, conf=0.45, verbose=False)]
        
        # 2. Bot Modeli Varsa Çalıştır
        if model_bot:
            results_list.append(model_bot(img, conf=0.50, verbose=False))

        violation_detected_in_frame = False
        detected_label = ""

        # Sonuçları Çizdir (KUTUCUKLAR BURADA ÇİZİLİYOR)
        for results in results_list:
            for r in results:
                for box in r.boxes:
                    # Koordinatları al
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Sınıf ismini al
                    cls = int(box.cls[0])
                    label = r.names[cls] if r.names and cls in r.names else "Unknown"
                    
                    # İhlal kontrolü
                    if label.startswith("NO-") or "Maks_Takmiyor" in label:
                        violation_detected_in_frame = True
                        detected_label = label

                    # Rengi belirle ve ÇİZ
                    color = colors.get(label, (255, 0, 255)) # Bulamazsa mor yap
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Yazıyı yaz
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(img, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 5 Saniye Kuralı ve Kayıt Mantığı
        if violation_detected_in_frame:
            if self.violation_start_time is None:
                self.violation_start_time = time.time()
                self.violation_logged = False
                self.current_violation_label = detected_label
            
            elapsed_time = time.time() - self.violation_start_time
            
            if elapsed_time < 5:
                # Geri sayım
                remaining = 5 - elapsed_time
                cv2.putText(img, f"IHLAL TESPIT EDILDI: {int(remaining)+1}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
            else:
                # 5 saniye doldu
                if not self.violation_logged:
                    cv2.putText(img, "KAYDEDILDI!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    # Veritabanına Kaydet
                    try:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Resmi byte formatına çevir
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        stream = io.BytesIO()
                        pil_img.save(stream, format='JPEG')
                        img_byte = stream.getvalue()
                        
                        # DB Bağlantısı
                        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
                        c = conn.cursor()
                        c.execute("INSERT INTO violations (timestamp, violation_type, image) VALUES (?, ?, ?)",
                                  (timestamp, self.current_violation_label, img_byte))
                        conn.commit()
                        conn.close()
                        
                        self.violation_logged = True
                    except Exception as e:
                        print(f"Kayıt hatası: {e}")
                else:
                    cv2.putText(img, "KAYIT VERITABANINDA MEVCUT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # İhlal yoksa sayacı sıfırla
            self.violation_start_time = None
            self.violation_logged = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- ARAYÜZ ---
st.sidebar.title("🔧 Sistem Ayarları")
mod = st.sidebar.radio("Mod Seç:", ["🎥 Saha Kamerası", "👷 Şef Paneli"])

if mod == "🎥 Saha Kamerası":
    st.title("🎥 Saha Denetim Ekranı")
    st.write("Sistem şu an aktif. İhlal durumunda kutucuklar kırmızı yanar.")
    
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

elif mod == "👷 Şef Paneli":
    st.title("👷 İhlal Kayıtları")
    
    if st.button("🔄 Yenile"):
        st.rerun()
        
    # Basit Silme Butonu (Tek Tek Silme Özelliği ile)
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            c = conn.cursor()
            c.execute("SELECT id, timestamp, violation_type, image FROM violations ORDER BY id DESC")
            rows = c.fetchall()
            conn.close()
            
            if not rows:
                st.info("Henüz kayıtlı ihlal yok.")
            else:
                for row in rows:
                    r_id, r_ts, r_type, r_img = row
                    with st.container(border=True):
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            try:
                                image = Image.open(io.BytesIO(r_img))
                                st.image(image, use_container_width=True)
                            except:
                                st.error("Resim açılamadı")
                        with c2:
                            st.error(f"🚨 {r_type}")
                            st.write(f"🕒 {r_ts}")
                            
                            # Tekil Silme Butonu
                            if st.button(f"🗑️ Sil (ID: {r_id})", key=f"del_{r_id}"):
                                try:
                                    conn_del = sqlite3.connect(DB_PATH, check_same_thread=False)
                                    c_del = conn_del.cursor()
                                    c_del.execute("DELETE FROM violations WHERE id=?", (r_id,))
                                    conn_del.commit()
                                    conn_del.close()
                                    st.success("Kayıt silindi.")
                                    time.sleep(0.5)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Silinemedi: {e}")
        except Exception as e:
            st.error(f"Veritabanı hatası: {e}")