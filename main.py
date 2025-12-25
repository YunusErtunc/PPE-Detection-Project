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

# --- VERİTABANI BAĞLANTISI ---
# Bağlantıyı globalde değil, fonksiyon içinde yöneteceğiz ki silerken sorun çıkmasın
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

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

# İlk açılışta tabloyu oluştur
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

# --- GÖRÜNTÜ İŞLEME SINIFI ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.violation_start_time = None
        self.violation_logged = False
        self.current_violation_label = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Modelleri çalıştır
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

        # 5 Saniye Mantığı
        if violation_detected_in_frame:
            if self.violation_start_time is None:
                self.violation_start_time = time.time()
                self.violation_logged = False
                self.current_violation_label = detected_label
            
            elapsed_time = time.time() - self.violation_start_time
            remaining_time = 5 - elapsed_time

            if elapsed_time < 5:
                text = f"DIKKAT! Ihlal Kaydediliyor: {int(remaining_time)+1}"
                cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            else:
                if not self.violation_logged:
                    cv2.putText(img, "VERITABANINA KAYDEDILDI!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    try:
                        # Görüntüyü byte'a çevir
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        stream = io.BytesIO()
                        pil_img.save(stream, format='JPEG')
                        img_byte = stream.getvalue()
                        
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Anlık bağlantı aç-kapa (En güvenli yöntem)
                        local_conn = get_db_connection()
                        local_c = local_conn.cursor()
                        local_c.execute("INSERT INTO violations (timestamp, violation_type, image) VALUES (?, ?, ?)",
                                        (timestamp, self.current_violation_label, img_byte))
                        local_conn.commit()
                        local_conn.close()
                        
                        print(f"🚨 KAYIT BAŞARILI: {self.current_violation_label}")
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
    st.write("Kamera, 5 saniye boyunca kesintisiz ihlal tespit ederse Şef Paneline düşer.")
    
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    webrtc_streamer(
        key="isg-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif mod == "👷 Şef Paneli (Admin)":
    st.title("👷 Şef Denetim Paneli")
    
    col_refresh, col_delete_all = st.columns([1, 4])
    
    with col_refresh:
        if st.button("🔄 Listeyi Yenile"):
            st.rerun()
            
    with col_delete_all:
        # --- GÜÇLENDİRİLMİŞ SİLME BUTONU ---
        if st.button("🗑️ HER ŞEYİ SİL VE SIFIRLA (KESİN ÇÖZÜM)", type="primary"):
            try:
                # 1. Streamlit Önbelleğini Temizle (Eski görüntülerin kalmasını engeller)
                st.cache_resource.clear()
                st.cache_data.clear()
                
                # 2. Klasörleri Sil
                if os.path.exists("runs"):
                    shutil.rmtree("runs", ignore_errors=True)
                if os.path.exists("yeni_veri"):
                    shutil.rmtree("yeni_veri", ignore_errors=True)
                
                # 3. Veritabanı Temizliği (Hem dosya silme hem SQL truncate denemesi)
                # Windows'ta dosya kilitli olabilir, bu yüzden try-except ile iki yöntem deniyoruz
                try:
                    # Yöntem A: Dosyayı direk sil
                    if os.path.exists(DB_PATH):
                        os.remove(DB_PATH)
                    init_db() # Yeni boş dosya oluştur
                except PermissionError:
                    # Yöntem B: Dosya kilitliyse SQL ile içini boşalt ve sayacı sıfırla
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute("DELETE FROM violations") # Verileri sil
                    # SQLite sayacını (ID) sıfırlamak için sqlite_sequence tablosunu temizle
                    c.execute("DELETE FROM sqlite_sequence WHERE name='violations'") 
                    conn.commit()
                    conn.close()

                st.success("Tüm sistem, önbellek ve veritabanı sıfırlandı!")
                time.sleep(2)
                st.rerun()
                
            except Exception as e:
                st.error(f"Hata: {e}")

    # Verileri Listele
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT id, timestamp, violation_type, image FROM violations ORDER BY id DESC")
        rows = c.fetchall()
        conn.close() # İş bitince kapat

        if not rows:
            st.info("Veritabanı boş. Kayıt bulunamadı. ✅")
        else:
            for row in rows:
                record_id, ts, v_type, img_data = row
                with st.container(border=True): 
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        try:
                            image = Image.open(io.BytesIO(img_data))
                            st.image(image, caption=f"ID: {record_id}", use_container_width=True)
                        except:
                            st.error("Görüntü hatası")
                    with col2:
                        st.error(f"🚨 {v_type}")
                        st.write(f"🕒 {ts}")
                        
                        if st.button(f"🗑️ Sil (ID: {record_id})", key=f"del_{record_id}"):
                            del_conn = get_db_connection()
                            del_c = del_conn.cursor()
                            del_c.execute("DELETE FROM violations WHERE id=?", (record_id,))
                            del_conn.commit()
                            del_conn.close()
                            st.rerun()
    except Exception as e:
        st.error(f"Veritabanı okunurken hata: {e}")