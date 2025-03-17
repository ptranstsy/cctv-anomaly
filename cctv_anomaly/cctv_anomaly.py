import cv2
import numpy as np
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from flask import Flask, Response, render_template

# Inisialisasi Flask
app = Flask(__name__)

# Load model YOLOv8
model = YOLO("yolov8n.pt")

# Inisialisasi Firebase
try:
    cred = credentials.Certificate(r"C:\Users\User\cctv_anomaly\serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    print("Firebase terhubung")
except Exception as e:
    print(f"Error Firebase: {e}")
db = firestore.client()

# Video source (gunakan global untuk cap)
video_source = 0  
global cap
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Error: Kamera atau video source tidak dapat dibuka. Coba ganti video_source.")
    exit()

# CamID
cam_id = "C001"

# Daftar objek berbahaya
dangerous_objects = {
    "knife": "High", "gun": "High", "person": "Low", "fire": "High",
    "suspicious_bag": "Medium", "fight": "High"
}

# Menyimpan status alert
detected_alerts = set()

# Fungsi untuk menghasilkan frame deteksi
def generate_frames():
    global cap  # Deklarasikan cap sebagai variabel global
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame, memulai ulang capture...")
            cap = cv2.VideoCapture(video_source)  # Coba restart capture
            continue

        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                confidence = box.conf[0].item()
                if label in dangerous_objects and confidence > 0.5:
                    priority = dangerous_objects[label]
                    color = (0, 0, 255) if priority == "High" else (0, 165, 255) if priority == "Medium" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} ({priority})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    alert_key = f"{label}_{cam_id}"
                    if alert_key not in detected_alerts:
                        detected_alerts.add(alert_key)
                        alert_data = {
                            "Level": priority,
                            "Description": label,
                            "Date": datetime.now().isoformat(),
                            "CamID": cam_id
                        }
                        db.collection("alerts").add(alert_data)
                        print(f"🔥 Alert terkirim: {alert_data}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route untuk halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Route untuk video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Server akan dimulai...")
    # Jalankan server Flask
    app.run(host='0.0.0.0', port=5000, threaded=True)
    # Tutup capture dan jendela saat server berhenti
    cap.release()
    cv2.destroyAllWindows()