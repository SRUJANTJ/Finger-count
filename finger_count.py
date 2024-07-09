import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # Allow detection of two hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    count = 0

    # Thumb
    if hand_landmarks[4][0] < hand_landmarks[3][0]:
        count += 1

    # Other four fingers
    for tip_id in [8, 12, 16, 20]:
        if hand_landmarks[tip_id][1] < hand_landmarks[tip_id - 2][1]:
            count += 1

    return count

output_file = open('finger_count.txt', 'w')

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    total_fingers_count = 0  # Initialize total fingers count

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert landmarks to a more convenient format
            hand_landmarks_list = []
            for lm in hand_landmarks.landmark:
                h, w, _ = img.shape
                hand_landmarks_list.append((int(lm.x * w), int(lm.y * h)))

            # Count fingers for this hand
            fingers_count = count_fingers(hand_landmarks_list)
            total_fingers_count += fingers_count  # Add to total fingers count

    # Display and write the total fingers count
    cv2.putText(img, f'Fingers: {total_fingers_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    output_file.write(f'Fingers: {total_fingers_count}\n')

    cv2.imshow('Finger Count', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

output_file.close()
cap.release()
cv2.destroyAllWindows()
