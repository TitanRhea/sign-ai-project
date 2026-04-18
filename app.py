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

# --- 2. Ο ΔΙΚΟΣ ΣΟΥ ΣΤΑΘΕΡΟΠΟΙΗΤΗΣ ---
current_candidate = None
consecutive_frames = 0
REQUIRED_FRAMES = 2          
CONFIDENCE_THRESHOLD = 0.40  
last_spoken_word = None
last_spoken_time = 0

pending_gesture = None
gesture_start_time = 0

@app.route('/')
def index():
    return "SignAI Brain is LIVE!"

@socketio.on('process_landmarks')
def handle_landmarks(data):
    global current_candidate, consecutive_frames, last_spoken_word, last_spoken_time
    global pending_gesture, gesture_start_time
    
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

    # =========================================================
    # --- Η ΤΕΛΕΙΑ ΛΟΓΙΚΗ ΤΗΣ ΟΛΓΑΣ ---
    # =========================================================
    wrist_x = raw_landmarks[0]['x']
    wrist_y = raw_landmarks[0]['y']
    
    index_up = raw_landmarks[8]['y'] < (raw_landmarks[5]['y'] - 0.04)
    middle_down = raw_landmarks[12]['y'] > raw_landmarks[10]['y']
    ring_down = raw_landmarks[16]['y'] > raw_landmarks[14]['y']
    pinky_down = raw_landmarks[20]['y'] > raw_landmarks[18]['y']
    
    is_fist_base = middle_down and ring_down and pinky_down
    is_high = wrist_y < 0.80 
    is_side = abs(wrist_x - 0.5) > 0.15 
    
    if active_now in ["efharisto", "kalimera", "kalo_mesimeri"]:
        if not is_side:
            # Κέντρο = Ευχαριστώ
            active_now = "efharisto"
            prediction_prob = 0.99
            pending_gesture = None
        else:
            # Στο πλάι (άρα Καλημέρα ή Καλό μεσημέρι)
            if is_high and is_fist_base:
                if pending_gesture is None:
                    pending_gesture = "checking"
                    gesture_start_time = time.time()
            
            if pending_gesture == "checking":
                if index_up:
                    # Σήκωσε δείκτη! -> Καλημέρα
                    active_now = "kalimera"
                    prediction_prob = 0.99
                    pending_gesture = None
                elif not is_high or not is_side:
                    # Το χέρι χάθηκε προς τα κάτω ή έφυγε, χωρίς να δούμε δείκτη! -> Καλό Μεσημέρι
                    if "kalo_mesimeri" != last_spoken_word or (time.time() - last_spoken_time > 2.0):
                        socketio.emit('new_sign', {'word': "kalo_mesimeri"})
                        last_spoken_word = "kalo_mesimeri"
                        last_spoken_time = time.time()
                    
                    pending_gesture = None
                    current_candidate = None
                    consecutive_frames = 0
                    return # Τερματισμός, το στείλαμε κατευθείαν στην οθόνη!
                else:
                    # Σιγή ασυρμάτου, περιμένουμε την επόμενη κίνηση
                    active_now = "noise"
                    prediction_prob = 0.0
                    
                    if (time.time() - gesture_start_time) > 1.5:
                        pending_gesture = None
            else:
                active_now = "noise"

    # ΕΙΔΙΚΟΣ ΚΑΝΟΝΑΣ ΓΙΑ ΓΕΙΑ (Παραμένει ίδιος)
    if active_now not in ["efharisto", "kalimera", "kalo_mesimeri"]:
        if prediction_prob < CONFIDENCE_THRESHOLD:
            if active_now == "geia" and prediction_prob >= 0.30:
                pass 
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
