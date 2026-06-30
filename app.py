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
side_memory_time = 0 

@app.route('/')
def index():
    return "SignAI Brain is LIVE!"

@socketio.on('process_landmarks')
def handle_landmarks(data):
    global current_candidate, consecutive_frames, last_spoken_word, last_spoken_time
    global side_memory_time
    
    if model is None: return
    
    raw_landmarks = data.get('landmarks')
    if not raw_landmarks or len(raw_landmarks) != 21: return
    
    # Μετατροπή Συντεταγμένων
    row = []
    base_x = raw_landmarks[0]['x']
    base_y = raw_landmarks[0]['y']
    for lm in raw_landmarks: row.append(lm['x'] - base_x)
    for lm in raw_landmarks: row.append(lm['y'] - base_y)

    probabilities = model.predict_proba([row])[0]
    best_class_index = np.argmax(probabilities)
    active_now = model.classes_[best_class_index]
    prediction_prob = probabilities[best_class_index]

    wrist_x = raw_landmarks[0]['x']
    wrist_y = raw_landmarks[0]['y']
    middle_tip = raw_landmarks[12]
    
    is_index_up = raw_landmarks[8]['y'] < (raw_landmarks[12]['y'] - 0.08)
    is_side = abs(wrist_x - 0.5) > 0.16 
    is_center = abs(wrist_x - 0.5) < 0.20
    is_high = wrist_y < 0.85
    # Πιο ευρύ όριο για το πηγούνι ώστε να αναγνωρίζεται η κίνηση με μεγαλύτερη ευκολία
    is_chin_level = 0.25 < wrist_y < 0.85 

    # Logikή Διαδρομής
    if is_high and is_side:
        side_memory_time = time.time()
        if is_index_up:
            active_now = "kalimera"
            prediction_prob = 0.99
            side_memory_time = 0 
        else:
            active_now = "noise"
            
    has_side_memory = (time.time() - side_memory_time) < 2.0
    
    # Αναγνώριση "Καλό μεσημέρι" (πρέπει να έχει πάει στο πηγούνι - κέντρο)
    if has_side_memory and is_center and is_chin_level: 
        active_now = "kalo_mesimeri"
        prediction_prob = 0.99

    if active_now not in ["efharisto", "kalimera", "kalo_mesimeri"]:
        if prediction_prob < CONFIDENCE_THRESHOLD:
            if active_now == "geia" and prediction_prob >= 0.30: pass 
            else: active_now = 'noise'

    if active_now:
        if active_now == current_candidate:
            consecutive_frames += 1
        else:
            current_candidate = active_now
            consecutive_frames = 1
            
        if consecutive_frames >= REQUIRED_FRAMES and current_candidate not in ['background', 'noise']:
            if current_candidate != last_spoken_word or (time.time() - last_spoken_time > 1.0):
                socketio.emit('new_sign', {'word': current_candidate})
                last_spoken_word = current_candidate
                last_spoken_time = time.time()
                
                # Ειδική διαχείριση απελευθέρωσης για το "καλό μεσημέρι"
                if current_candidate != "kalo_mesimeri":
                    consecutive_frames = 0
                    current_candidate = None 
                    side_memory_time = 0
                else:
                    consecutive_frames = 0
    else:
        consecutive_frames = 0
        current_candidate = None

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    socketio.run(app, host='0.0.0.0', port=port)
