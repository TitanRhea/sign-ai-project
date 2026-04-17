import pickle
import numpy as np
import time
from flask import Flask
from flask_socketio import SocketIO
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 1. ΦΟΡΤΩΣΗ ΤΟΥ ΕΓΚΕΦΑΛΟΥ
try:
    with open('sign_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Το μοντέλο φορτώθηκε!")
except FileNotFoundError:
    print("❌ Σφάλμα: Δεν βρέθηκε το sign_model.pkl!")
    model = None

@app.route('/')
def index():
    return "SignAI Brain is LIVE!"

# 2. ΟΤΑΝ ΠΑΙΡΝΕΙ ΣΥΝΤΕΤΑΓΜΕΝΕΣ ΑΠΟ ΤΟΝ CHROME
@socketio.on('process_landmarks')
def handle_landmarks(data):
    if model is None: return
    
    raw_landmarks = data.get('landmarks')
    if not raw_landmarks or len(raw_landmarks) != 21: return
    
    # Μετατροπή των σημείων ακριβώς όπως τα έμαθε το AI σου
    row = []
    base_x = raw_landmarks[0]['x']
    base_y = raw_landmarks[0]['y']
    
    for lm in raw_landmarks: row.append(lm['x'] - base_x)
    for lm in raw_landmarks: row.append(lm['y'] - base_y)

    # Πρόβλεψη Λέξης
    probabilities = model.predict_proba([row])[0]
    best_class_index = np.argmax(probabilities)
    active_now = model.classes_[best_class_index]
    prediction_prob = probabilities[best_class_index]

    # Δικλείδα για Καλό Μεσημέρι
    wrist_y = raw_landmarks[0]['y']
    middle_finger_tip_y = raw_landmarks[12]['y']
    
    is_hand_vertical = middle_finger_tip_y < wrist_y - 0.1
    is_hand_at_chin = wrist_y < 0.65
    
    if active_now == "efharisto" and is_hand_at_chin and is_hand_vertical:
        active_now = "kalo_mesimeri"
        prediction_prob = max(prediction_prob, 0.85)

    # Αν είναι σίγουρο, στέλνει τη λέξη πίσω στο Αβατάρ
    if prediction_prob >= 0.50 and active_now not in ['background', 'noise']:
        socketio.emit('new_sign', {'word': active_now})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    socketio.run(app, host='0.0.0.0', port=port)
