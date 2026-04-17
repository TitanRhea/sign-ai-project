import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
from flask import Flask
from flask_socketio import SocketIO
import threading

# --- ΡΥΘΜΙΣΕΙΣ SERVER ---
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- ΦΟΡΤΩΣΗ ΤΟΥ ΕΓΚΕΦΑΛΟΥ ---
try:
    with open('sign_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("❌ Σφάλμα: Δεν βρέθηκε το sign_model.pkl στον φάκελο!")
    exit()

# --- MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- ΟΙ ΡΥΘΜΙΣΕΙΣ ΣΟΥ (3 Frames / 50% Confidence) ---
REQUIRED_FRAMES = 3
CONFIDENCE_THRESHOLD = 0.50

def recognition_logic():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    current_candidate = None
    consecutive_frames = 0
    last_spoken_word = None
    last_spoken_time = 0

    print("🚀 SignAI Server: Η κάμερα και η αναγνώριση ξεκίνησαν...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        active_now = None
        prediction_prob = 0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Σημεία για τη δικλείδα "Καλό Μεσημέρι"
            wrist = hand_landmarks.landmark[0]
            middle_finger_tip = hand_landmarks.landmark[12]
            
            row = []
            base_x, base_y = wrist.x, wrist.y

            for lm in hand_landmarks.landmark:
                row.append(lm.x - base_x)
            for lm in hand_landmarks.landmark:
                row.append(lm.y - base_y)

            # Πρόβλεψη
            probabilities = model.predict_proba([row])[0]
            best_class_index = np.argmax(probabilities)
            active_now = model.classes_[best_class_index]
            prediction_prob = probabilities[best_class_index]

            # 🔒 Η ΔΙΚΛΕΙΔΑ ΣΟΥ ΓΙΑ ΤΟ ΚΑΛΟ ΜΕΣΗΜΕΡΙ
            is_hand_vertical = middle_finger_tip.y < wrist.y - 0.1
            is_hand_at_chin = wrist.y < 0.65
            if active_now == "efharisto" and is_hand_at_chin and is_hand_vertical:
                active_now = "kalo_mesimeri"
                prediction_prob = max(prediction_prob, 0.85)

        # Φιλτράρισμα θορύβου
        if prediction_prob < CONFIDENCE_THRESHOLD:
            active_now = 'noise'

        # Σταθεροποιητής (4 Frames Lock)
        if active_now:
            if active_now == current_candidate:
                consecutive_frames += 1
            else:
                current_candidate = active_now
                consecutive_frames = 1
            
            if consecutive_frames >= REQUIRED_FRAMES and current_candidate not in ['background', 'noise']:
                if current_candidate != last_spoken_word or (time.time() - last_spoken_time > 2.0):
                    print(f"📡 Λέξη: {current_candidate}")
                    
                    # --- Η ΓΕΦΥΡΑ: Στέλνουμε τη λέξη στο camera.html ---
                    socketio.emit('new_sign', {'word': current_candidate})
                    
                    last_spoken_word = current_candidate
                    last_spoken_time = time.time()
                    consecutive_frames = 0
        else:
            consecutive_frames = 0

        time.sleep(0.01)

# Ξεκινάμε την αναγνώριση σε Background Thread
threading.Thread(target=recognition_logic, daemon=True).start()

@app.route('/')
def index():
    return "<h1>Ο Python Server της SignAI τρέχει!</h1><p>Τώρα άνοιξε το camera.html στον Chrome.</p>"

if __name__ == '__main__':
    # Ο server τρέχει στο localhost:5000
    socketio.run(app, port=5000, debug=False)