import cv2
import mediapipe as mp
import serial

# Replace 'COMx' with the COM port of your Bluetooth module
bluetooth = serial.Serial('/dev/tty.HC-05', 9600)  # Update with your Bluetooth COM port

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Gesture recognition function
def recognize_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Calculate whether each finger is up or down
    index_up = index_tip.y < landmarks[6].y
    middle_up = middle_tip.y < landmarks[10].y
    ring_up = ring_tip.y < landmarks[14].y
    pinky_up = pinky_tip.y < landmarks[18].y
    thumb_up = thumb_tip.y < landmarks[3].y

    # Determine the number of fingers up (excluding thumb)
    fingers_up = sum([index_up, middle_up, ring_up, pinky_up])

    if fingers_up == 1 and index_up and not middle_up and not ring_up and not pinky_up:
        return 'F'  # Forward
    elif fingers_up == 2 and index_up and middle_up and not ring_up and not pinky_up:
        return 'R'  # Right
    elif fingers_up == 3 and index_up and middle_up and ring_up and not pinky_up:
        return 'L'  # Left
    elif fingers_up == 4 and index_up and middle_up and ring_up and pinky_up:
        return 'B'  # Backward
    else:
        return 'S'  # Stop

# Start capturing video
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize gesture
            gesture = recognize_gesture(hand_landmarks.landmark)
            print(gesture)

            # Send gesture command to Bluetooth
            bluetooth.write(gesture.encode())

    cv2.imshow('Gesture Controlled Buggy', frame)
    # if cv2.waitKey(5) & 0xFF == 27:
    #     break
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
bluetooth.close()