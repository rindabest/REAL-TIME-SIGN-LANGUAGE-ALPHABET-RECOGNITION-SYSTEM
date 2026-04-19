"""
ASL (American Sign Language) Recognition - Webcam Demo
======================================================
Script nhan dang ngon ngu ky hieu qua webcam su dung:
  - MediaPipe Hands: phat hien va trich xuat toa do ban tay
  - Random Forest (RD.pkl): mo hinh da huan luyen de phan loai chu cai

Cach chay:
    python test_cam.py

Yeu cau:
    - File RD.pkl cung thu muc
    - Cai dat: pip install opencv-python mediapipe joblib numpy scikit-learn
    
Nhan 'q' de thoat.
"""

import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import sys

# ===================== CAU HINH =====================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RD.pkl')
CAMERA_INDEX = 0            # Index cua webcam (0 = webcam mac dinh)
MAX_NUM_HANDS = 1           # So tay toi da can phat hien
MIN_DETECTION_CONF = 0.7    # Nguong tin cay phat hien tay
MIN_TRACKING_CONF = 0.5     # Nguong tin cay tracking tay
WINDOW_NAME = 'ASL Recognition - Nhan dang chu cai'

# ===================== TAI MODEL =====================
print("=" * 50)
print("  ASL RECOGNITION - WEBCAM DEMO")
print("=" * 50)

if not os.path.exists(MODEL_PATH):
    print(f"\n[ERROR] Khong tim thay file model tai: {MODEL_PATH}")
    print("Vui long dam bao file RD.pkl nam cung thu muc voi script nay.")
    sys.exit(1)

print(f"\n[INFO] Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("[INFO] Model loaded successfully!")

# ===================== KHOI TAO MEDIAPIPE =====================
print("[INFO] Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_NUM_HANDS,
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=MIN_TRACKING_CONF
)
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
print("[INFO] MediaPipe initialized!")

# ===================== MO CAMERA =====================
print(f"[INFO] Opening Camera (index={CAMERA_INDEX})...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"\n[ERROR] Cannot open camera index={CAMERA_INDEX}.")
    print("Please check:")
    print("  - Is webcam connected?")
    print("  - Is another app using the webcam?")
    print("  - Try changing CAMERA_INDEX (0, 1, 2, ...)")
    sys.exit(1)

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Camera ready!")
print("[INFO] Press 'q' to quit.\n")

# ===================== MAIN LOOP =====================
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("[WARNING] Cannot read frame from camera. Retrying...")
        continue

    # Flip image (mirror) for natural experience
    image = cv2.flip(image, 1)
    
    # Get frame dimensions
    h, w, _ = image.shape
    
    # Convert to RGB (MediaPipe requires RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Optimization: mark image as not writeable to speed up
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    # Draw guide text
    cv2.putText(image, "Dua tay vao camera de nhan dang", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(image, "Nhan 'q' de thoat", (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on hand
            mp_draw.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract landmarks - SAME method as training
            # Using relative coordinates from landmark 0 (wrist)
            lm_list = []
            x0 = hand_landmarks.landmark[0].x
            y0 = hand_landmarks.landmark[0].y
            for lm in hand_landmarks.landmark:
                lm_list.append(lm.x - x0)
                lm_list.append(lm.y - y0)

            # Predict letter
            try:
                prediction = model.predict([lm_list])
                label = prediction[0]
                
                # Get confidence probability
                proba = model.predict_proba([lm_list])
                confidence = np.max(proba) * 100

                # Draw result on screen
                # Background box for text
                cv2.rectangle(image, (30, 10), (400, 90), (0, 0, 0), -1)
                cv2.rectangle(image, (30, 10), (400, 90), (0, 255, 0), 2)
                
                # Recognized letter
                display_label = label
                if label == 'del':
                    display_label = 'DELETE'
                elif label == 'space':
                    display_label = 'SPACE'
                elif label == 'nothing':
                    display_label = 'NOTHING'
                    
                cv2.putText(image, f"Letter: {display_label}", (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(image, f"Confidence: {confidence:.1f}%", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)

            except Exception as e:
                cv2.putText(image, f"Error: {str(e)}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # When no hand detected
        cv2.rectangle(image, (30, 10), (400, 50), (0, 0, 0), -1)
        cv2.putText(image, "No hand detected", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # Display image
    cv2.imshow(WINDOW_NAME, image)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n[INFO] Exiting...")
        break

# ===================== CLEANUP =====================
cap.release()
cv2.destroyAllWindows()
hands.close()
print("[INFO] Camera closed and resources released.")
print("[INFO] Goodbye!")
