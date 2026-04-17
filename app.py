import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
from flask import Flask
from flask_socketio import SocketIO
import threading
import os

# ΕΙΔΙΚΗ ΔΙΟΡΘΩΣΗ ΓΙΑ RENDER (Mediapipe Linux Fix)
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_draw

# Ρυθμίσεις Flask & SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ΦΟΡΤΩΣΗ ΤΟΥ ΜΟΝΤΕΛΟΥ
try:
    with open('sign_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("❌ Σφάλμα: Δεν βρέθηκε το sign_model.pkl!")
    exit()

# Ρυθμίσεις Mediapipe
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Ρυθμίσεις Αναγνώρισης (3 Frames / 50%)
REQUIRED_FRAMES = 3
CONFIDENCE_THRESHOLD = 0.50

def recognition_logic():
    # Στο Render/Cloud δεν έχουμε πραγματική κάμερα, αλλά ο κώδικας 
    # πρέπει να τρέχει χωρίς να κρασάρει αν δεν βρει συσκευή.
    cap = cv2.VideoCapture(0)
    current_candidate = None
    consecutive_frames = 0
    last_spoken_word = None
    last_spoken_time = 0

    print("🚀 SignAI Server: Started")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1) # Περιμένει αν δεν υπάρχει κάμερα
            continue
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        active_now = None
        prediction_prob = 0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            wrist = hand_landmarks.landmark[0]
            middle_finger_tip = hand_landmarks.landmark[12]
            
            row = []
            base_x, base_y = wrist.x, wrist.y
            for lm in hand_landmarks.landmark:
                row.append(lm.x - base_x)
            for lm in hand_landmarks.landmark:
                row.append(lm.y - base_y)

            probabilities = model.predict_proba([row])[0]
            best_class_index = np.argmax(probabilities)
            active_now = model.classes_[best_class_index]
            prediction_prob = probabilities[best_class_index]

            # Δικλείδα για Καλό Μεσημέρι
            is_hand_vertical = middle_finger_tip.y < wrist.y - 0.1
            is_hand_at_chin = wrist.y < 0.65
            if active_now == "efharisto" and is_hand_at_chin and is_hand_vertical:
                active_now = "kalo_mesimeri"
                prediction_prob = max(prediction_prob, 0.85)

        if prediction_prob < CONFIDENCE_THRESHOLD:
            active_now = 'noise'

        if active_now:
            if active_now == current_candidate:
                consecutive_frames += 1
            else:
                current_candidate = active_now
                consecutive_frames = 1
            
            if consecutive_frames >= REQUIRED_FRAMES and current_candidate not in ['background', 'noise']:
                if current_candidate != last_spoken_word or (time.time() - last_spoken_time > 2.0):
                    print(f"📡 Emission: {current_candidate}")
                    socketio.emit('new_sign', {'word': current_candidate})
                    last_spoken_word = current_candidate
                    last_spoken_time = time.time()
                    consecutive_frames = 0
        else:
            consecutive_frames = 0
        time.sleep(0.01)

threading.Thread(target=recognition_logic, daemon=True).start()

@app.route('/')
def index():
    return "SignAI Server is LIVE"

if __name__ == '__main__':
    # ΔΙΟΡΘΩΣΗ ΠΟΡΤΑΣ ΓΙΑ ΤΟ RENDER
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port)
