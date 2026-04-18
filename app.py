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
REQUIRED_FRAMES = 3 
CONFIDENCE_THRESHOLD = 0.50 
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

    # --- Η ΝΕΑ, ΑΠΟΛΥΤΑ ΧΑΛΑΡΗ ΔΙΚΛΕΙΔΑ ΓΙΑ ΤΟ ΚΑΛΟ ΜΕΣΗΜΕΡΙ ---
    wrist_y = raw_landmarks[0]['y']
    middle_finger_tip_y = raw_landmarks[12]['y']
    
    # Αρκεί η άκρη του δαχτύλου να είναι πιο πάνω από τον καρπό (δεν απαιτείται τέλεια καθετότητα)
    is_hand_vertical = middle_finger_tip_y < wrist_y 
    # Αρκεί το χέρι να φαίνεται στην κάμερα (όριο στο 95% της οθόνης)
    is_hand_at_chin = wrist_y < 0.95 
    
    if active_now == "efharisto" and is_hand_at_chin and is_hand_vertical:
        active_now = "kalo_mesimeri"
        prediction_prob = max(prediction_prob, 0.85)

    # ΕΙΔΙΚΟΣ ΚΑΝΟΝΑΣ ΓΙΑ ΤΟ ΓΕΙΑ
    if prediction_prob < CONFIDENCE_THRESHOLD:
        if active_now == "geia" and prediction_prob >= 0.35:
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
