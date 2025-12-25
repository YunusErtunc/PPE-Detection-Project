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
import shutil

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="İSG Takip Sistemi", page_icon="🏗️", layout="wide")

# --- DOSYA YOLLARI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BARET_PATH = os.path.join(BASE_DIR, "models", "best.pt")
MODEL_BOT_PATH = os.path.join(BASE_DIR, "models", "bot.pt")
DB_PATH = os.path.join(BASE_DIR, "isg_database.db")

# --- VERİTABANI YARDIMCISI ---
def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS violations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  violation_type TEXT,
                  image BLOB)''')
    conn.commit()
    conn.close()

# Uygulama başladığında DB var mı kontrol et
if not os.path.exists(DB_PATH):
    init_db()

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

# --- VİDEO İŞLEME ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.violation_start_time = None
        self.violation_logged = False
        self.current_violation_label = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        results_list = [model_baret(img, conf=0.45, verbose=False)]
        if model_bot:
            results_list.append(model_bot(img, conf=0.50, verbose=False))

        violation_detected_in_frame = False
        detected_label = ""

        for results in results_list:
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = r.names[cls] if r.names and cls in r.names else "Unknown"
                    
                    if label.startswith("NO-") or "Maks_Takmiyor" in label:
                        violation_detected_in_frame = True
                        detected_label = label

                    color = colors.get(label, (255, 0, 255))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if violation_detected_in_frame:
            if self.violation_start_time is None:
                self.violation_start_time = time.time()
                self.violation_logged = False
                self.current_violation_label = detected_label
            
            elapsed_time = time.time() - self.violation_start_time
            if elapsed_time < 5:
                remaining = 5 - elapsed_time
                cv2.putText(img, f"DIKKAT! Ihlal Kaydediliyor: {int(remaining)+1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            else:
                if not self.violation_logged:
                    cv2.putText(img, "VERITABANINA KAYDEDILDI!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    try:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        stream = io.BytesIO()
                        pil_img.save(stream, format='JPEG')
                        img_byte = stream.getvalue()
                        
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Anlık Bağlantı Aç-Kapa (Çakışmayı önlemek için)
                        conn_local = get_db_connection()
                        c_local = conn_local.cursor()
                        c_local.execute("INSERT INTO violations (timestamp, violation_type, image) VALUES (?, ?, ?)",
                                        (timestamp, self.current_violation_label, img_byte))
                        conn_local.commit()
                        conn_local.close()
                        
                        self.violation_logged = True
                    except Exception as e:
                        print(f"Kayıt Hatası: {e}")
                else:
                    cv2.putText(img, "IHLAL DEVAM EDIYOR (KAYITLI)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            self.violation_start_time = None
            self.violation_logged = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- ARAYÜZ ---
st.sidebar.title("🔧 Sistem Ayarları")
mod = st.sidebar.radio("Giriş Modu Seçin:", ["🎥 Saha Kamerası", "👷 Şef Paneli (Admin)"])

if mod == "🎥 Saha Kamerası":
    st.title("🎥 Saha Denetim Modu")
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_streamer(key="isg-camera", mode=WebRtcMode.SENDRECV, rtc_configuration=rtc_configuration, video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False}, async_processing=True)

elif mod == "👷 Şef Paneli (Admin)":
    st.title("👷 Şef Denetim Paneli")
    
    col_refresh, col_delete_all = st.columns([1, 4])
    
    with col_refresh:
        if st.button("🔄 Yenile"):
            st.rerun()
            
    with col_delete_all:
        # --- ZORLA SİLME ALANI ---
        if st.button("🗑️ HER ŞEYİ SİL (ZORLA)", type="primary"):
            status_container = st.empty()
            status_container.warning("Silme işlemi başlatılıyor...")
            
            # 1. Streamlit Önbelleklerini Temizle
            st.cache_data.clear()
            st.cache_resource.clear()
            
            # 2. Dosya Sistemini Temizle
            if os.path.exists("runs"):
                shutil.rmtree("runs", ignore_errors=True)
            
            # 3. Veritabanını Fiziksel Olarak Silmeye Çalış
            db_deleted = False
            try:
                if os.path.exists(DB_PATH):
                    os.remove(DB_PATH) # Dosyayı direk sil
                    db_deleted = True
            except Exception as e:
                # Dosya kilitliyse SQL ile içini boşalt
                try:
                    conn_del = sqlite3.connect(DB_PATH)
                    c_del = conn_del.cursor()
                    c_del.execute("DELETE FROM violations")
                    c_del.execute("DELETE FROM sqlite_sequence WHERE name='violations'")
                    conn_del.commit()
                    conn_del.close()
                    c_del.execute("VACUUM") # Dosya boyutunu küçült ve diske yazmaya zorla
                    db_deleted = True
                except:
                    pass
            
            # 4. Yeniden Oluştur
            init_db()
            
            status_container.success("Sistem sıfırlandı! Lütfen sayfayı tamamen yenileyin.")
            time.sleep(2)
            st.rerun()

    # Verileri Göster
    if os.path.exists(DB_PATH):
        try:
            conn_read = get_db_connection()
            c_read = conn_read.cursor()
            c_read.execute("SELECT id, timestamp, violation_type, image FROM violations ORDER BY id DESC")
            rows = c_read.fetchall()
            conn_read.close()

            if not rows:
                st.info("Kayıt yok. (Silme işleminden sonra burayı görüyorsanız işlem başarılıdır.)")
            else:
                for row in rows:
                    r_id, r_ts, r_type, r_img = row
                    with st.container(border=True):
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            try:
                                st.image(Image.open(io.BytesIO(r_img)), caption=f"ID: {r_id}", use_container_width=True)
                            except: st.write("Resim hatası")
                        with c2:
                            st.error(r_type)
                            st.write(r_ts)
                            # Tekil silme
                            if st.button("Sil", key=f"d_{r_id}"):
                                cc = get_db_connection()
                                cc.execute("DELETE FROM violations WHERE id=?", (r_id,))
                                cc.commit()
                                cc.close()
                                st.rerun()
        except:
            st.error("Veritabanı okunurken hata oluştu veya dosya kilitli.")
    else:
        st.info("Veritabanı dosyası yok. Yeni kayıt bekleniyor.")