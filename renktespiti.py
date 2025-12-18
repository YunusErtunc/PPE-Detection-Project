import cv2
from ultralytics import YOLO

# 1. Modeli Yükle (İlk çalışmada internetten otomatik indirir)
# 'yolov8n.pt' en hızlı ve hafif modeldir.
model = YOLO("yolov8n.pt") 

# 2. Kamerayı Aç (0 genellikle bilgisayarın kendi kamerasını temsil eder)
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Genişlik
cap.set(4, 720)  # Yükseklik

while True:
    success, img = cap.read()
    
    if success:
        # YOLO ile tahmin yap
        results = model(img, stream=True)

        # Sonuçları ekrana çiz
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Koordinatları al
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Kutuyu çiz (OpenCV ile)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Görüntüyü göster
        cv2.imshow("ISG Projesi Test", img)

        # 'q' tuşuna basınca çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()