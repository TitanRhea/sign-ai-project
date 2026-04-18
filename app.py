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

# --- 2. ΣΤΑΘΕΡΟΠΟΙΗΤΗΣ ---
current_candidate = None
consecutive_frames = 0
REQUIRED_FRAMES = 2          
CONFIDENCE_THRESHOLD = 0.40  
last_spoken_word = None
last_spoken_time = 0

# ΜΕΤΑΒΛΗΤΕΣ ΓΙΑ ΤΗ ΔΙΑΔΡΟΜΗ ΤΗΣ ΟΛΓΑΣ
waiting_at_side = False # Έχει πάει το χέρι στο πλάι;

@app.route('/')
def index():
    return "SignAI Brain is LIVE!"

@socketio.on('process_landmarks')
def handle_landmarks(data):
    global current_candidate, consecutive_frames, last_spoken_word, last_spoken_time
    global waiting_at_side
    
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

    # --- ΓΕΩΜΕΤΡΙΚΟΣ ΕΛΕΓΧΟΣ ---
    wrist_x = raw_landmarks[0]['x']
    wrist_y = raw_landmarks[0]['y']
    middle_tip = raw_landmarks[12]
    
    # Έλεγχος Δείκτη
    is_index_up = raw_landmarks[8]['y'] < (raw_landmarks[12]['y'] - 0.08)
    
    # Έλεγχος Θέσης
    is_side = abs(wrist_x - 0.5) > 0.18  # Πολύ καθαρά στο πλάι
    is_center = abs(wrist_x - 0.5) < 0.15 # Στο κέντρο (μπροστά από το πρόσωπο)
    is_high = wrist_y < 0.75             # Ύψος ώμου και πάνω
    is_chin_level = 0.4 < wrist_y < 0.7   # Ακριβώς στο ύψος του πηγουνιού
    
    # Έλεγχος αν η παλάμη είναι οριζόντια (για το πηγούνι)
    dx = abs(middle_tip['x'] - raw_landmarks[0]['x'])
    dy = abs(middle_tip['y'] - raw_landmarks[0]['y'])
    is_horizontal = dx > dy

    # --- Η ΛΟΓΙΚΗ ΤΗΣ ΔΙΑΔΡΟΜΗΣ ---
    
    if is_high and is_side:
        # ΒΗΜΑ 1: Το χέρι είναι στο πλάι. Περιμένουμε...
        waiting_at_side = True
        
        if is_index_up:
            # Αν όσο είναι στο πλάι σηκωθεί ο δείκτης -> ΚΑΛΗΜΕΡΑ
            active_now = "kalimera"
            prediction_prob = 0.99
            waiting_at_side = False # Η διαδρομή ολοκληρώθηκε
        else:
            # Είναι στο πλάι αλλά γροθιά. Μην λες τίποτα ακόμα.
            active_now = "noise"
            
    elif waiting_at_side and is_center and is_chin_level and is_horizontal:
        # ΒΗΜΑ 2: Το χέρι ήταν στο πλάι και ΤΩΡΑ ήρθε στο πηγούνι οριζόντια -> ΚΑΛΟ ΜΕΣΗΜΕΡΙ
        active_now = "kalo_mesimeri"
        prediction_prob = 0.99
        waiting_at_side = False # Η διαδρομή ολοκληρώθηκε
        
    elif not is_high:
        # Αν το χέρι πέσει κάτω, μηδενίζουμε τη διαδρομή
        waiting_at_side = False

    # ΕΙΔΙΚΟΣ ΚΑΝΟΝΑΣ ΓΙΑ ΓΕΙΑ (Κλειδωμένος)
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
