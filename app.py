import pickle
import numpy as np
import time
import os
from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- 1. ΦΟΡΤΩΣΗ ΤΟΥ ΕΓΚΕΦΑΛΟΥ ---
try:
    with open('sign_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Το μοντέλο φορτώθηκε!")
except FileNotFoundError:
    print("❌ Σφάλμα: Δεν βρέθηκε το sign_model.pkl!")
    model = None

# --- 2. Ο ΔΙΚΟΣ ΣΟΥ ΣΤΑΘΕΡΟΠΟΙΗΤΗΣ (ΠΙΟ ΓΡΗΓΟΡΟΣ ΚΑΙ ΧΑΛΑΡΟΣ) ---
current_candidate = None
consecutive_frames = 0
REQUIRED_FRAMES = 2          # ΑΛΛΑΓΗ 1: Μόνο 2 frames για αστραπιαία αναγνώριση
CONFIDENCE_THRESHOLD = 0.40  # ΑΛΛΑΓΗ 2: Χαμηλώσαμε το γενικό όριο στο 40%
last_spoken_word = None
last_spoken_time = 0

@app.route('/')
def index():
    return "SignAI Brain is LIVE!"

@socketio.on('process_landmarks')
def handle_landmarks(data):
    global current_candidate, consecutive_frames, last_spoken_word, last_spoken_time
    
    if model is None: return
    
    raw_landmarks = data.get('landmarks')
    if not raw_landmarks or len(raw_landmarks) != 21: return
    
    # Μετατροπή Συντεταγμένων
    row = []
    base_x = raw_landmarks[0]['x']
    base_y = raw_landmarks[0]['y']
    
    for lm in raw_landmarks: row.append(lm['x'] - base_x)
    for lm in raw_landmarks: row.append(lm['y'] - base_y)

    # Πρόβλεψη από το AI
    probabilities = model.predict_proba([row])[0]
    best_class_index = np.argmax(probabilities)
    active_now = model.classes_[best_class_index]
    prediction_prob = probabilities[best_class_index]

    # Η ΔΙΚΛΕΙΔΑ ΓΙΑ ΤΟ ΚΑΛΟ ΜΕΣΗΜΕΡΙ (Οριζόντιο χέρι)
    wrist = raw_landmarks[0]
    middle_tip = raw_landmarks[12]
    
    dx = abs(middle_tip['x'] - wrist['x'])
    dy = abs(middle_tip['y'] - wrist['y'])
    
    is_hand_horizontal = dx > dy and dx > 0.05
    is_hand_at_chin = wrist['y'] < 0.75 
    
    if active_now == "efharisto" and is_hand_at_chin and is_hand_horizontal:
        active_now = "kalo_mesimeri"
        prediction_prob = max(prediction_prob, 0.85)

    # ΕΙΔΙΚΟΣ ΚΑΝΟΝΑΣ: Ακόμα πιο εύκολο για "Γεια" και "Καλημέρα"
    if prediction_prob < CONFIDENCE_THRESHOLD:
        if active_now in ["geia", "kalimera"] and prediction_prob >= 0.30:
            pass # Το αφήνουμε να περάσει με μόλις 30% σιγουριά
        else:
            active_now = 'noise'

    # --- Ο ΣΤΑΘΕΡΟΠΟΙΗΤΗΣ ---
    if active_now:
        if active_now == current_candidate:
            consecutive_frames += 1
        else:
            current_candidate = active_now
            consecutive_frames = 1
            
        if consecutive_frames >= REQUIRED_FRAMES and current_candidate not in ['background', 'noise']:
            if current_candidate != last_spoken_word or (time.time() - last_spoken_time > 2.0):
                socketio.emit('new_sign', {'word': current_candidate})
                last_spoken_word = current_candidate
                last_spoken_time = time.time()
                consecutive_frames = 0
    else:
        consecutive_frames = 0
        current_candidate = None

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    socketio.run(app, host='0.0.0.0', port=port)
