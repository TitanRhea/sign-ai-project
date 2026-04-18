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

# Μεταβλητές Μηχανής Καταστάσεων
pending_gesture = None
gesture_start_time = 0
locked_word = None

@app.route('/')
def index():
    return "SignAI Brain is LIVE!"

@socketio.on('process_landmarks')
def handle_landmarks(data):
    global current_candidate, consecutive_frames, last_spoken_word, last_spoken_time
    global pending_gesture, gesture_start_time, locked_word
    
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
    # --- Η ΛΟΓΙΚΗ ΤΗΣ ΟΛΓΑΣ: ΘΕΣΗ ΣΤΗΝ ΟΘΟΝΗ ΚΑΙ ΔΕΙΚΤΗΣ ---
    # =========================================================
    wrist_x = raw_landmarks[0]['x']
    wrist_y = raw_landmarks[0]['y']
    
    # Θέση: Είναι ψηλά; Είναι στο πλάι (μακριά από το 0.5 του κέντρου);
    is_high = wrist_y < 0.80
    is_side = abs(wrist_x - 0.5) > 0.15 
    
    # Δείκτης: Η άκρη του (8) είναι αισθητά πιο ψηλά από τη βάση του (5);
    is_index_up = raw_landmarks[8]['y'] < (raw_landmarks[5]['y'] - 0.04)

    # Παίρνουμε τον έλεγχο ΑΝ το AI βγάλει λέξη που συγχέεται
    if active_now in ["efharisto", "kalimera", "kalo_mesimeri"]:
        if is_high and is_side:
            # Είναι στο πλάι! Άρα είναι Καλημέρα ή Καλό Μεσημέρι. Ξεκινάμε το χρονόμετρο.
            if pending_gesture is None:
                pending_gesture = "waiting"
                gesture_start_time = time.time()
                active_now = "noise" # Φίμωση AI
            elif pending_gesture == "waiting":
                if (time.time() - gesture_start_time) < 0.4:
                    active_now = "noise" # Παραμένει φιμωμένο για 400ms
                else:
                    # Τέλος χρόνου! Αποφασίζουμε και κλειδώνουμε.
                    pending_gesture = "locked"
                    locked_word = "kalimera" if is_index_up else "kalo_mesimeri"
                    active_now = locked_word
                    prediction_prob = 0.99
            elif pending_gesture == "locked":
                # Αν ήταν κλειδωμένο στο Καλό Μεσημέρι αλλά σηκωθεί ο δείκτης, το κάνουμε Καλημέρα
                if locked_word == "kalo_mesimeri" and is_index_up:
                    locked_word = "kalimera"
                active_now = locked_word
                prediction_prob = 0.99
        else:
            # Είναι στο Κέντρο (Στέρνο)! Άρα είναι 100% Ευχαριστώ.
            pending_gesture = None
            locked_word = None
            active_now = "efharisto"
            prediction_prob = 0.99
    else:
        # Καθαρισμός καταστάσεων αν πέσουμε σε άλλη λέξη
        pending_gesture = None
        locked_word = None

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
