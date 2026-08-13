import cv2
import mediapipe as mp
import numpy as np
import os


label = "stop"    
samples_to_collect = 2000
save_dir = f"dataset/{label}"

# Create folder
os.makedirs(save_dir, exist_ok=True)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ========== CAMERA ==========
cap = cv2.VideoCapture(0)

count = 0

print(f"Collecting data for: {label}")
print("Press 'S' to start collecting")
print("Press 'Q' to quit")

collecting = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            points = []

            for lm in hand_landmarks.landmark:
                points.extend([lm.x, lm.y, lm.z])

            if collecting and count < samples_to_collect:
                np.save(f"{save_dir}/{count}.npy", points)
                count += 1

                cv2.putText(frame, f"Saved: {count}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display instructions
    cv2.putText(frame, f"Label: {label}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if not collecting:
        cv2.putText(frame, "Press S to Start", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Dataset Collector", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        collecting = True

    elif key == ord('q'):
        break

    if count >= samples_to_collect:
        print(f"Finished collecting {samples_to_collect} samples for {label}")
        break

cap.release()
cv2.destroyAllWindows()