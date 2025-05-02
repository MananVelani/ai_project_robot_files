import cv2
import mediapipe as mp
import math
import socket

robotAddressPort = ("192.168.137.10", 12345)

# initialize mediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# initialize videoCapture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Tracking', 1280, 720)

def sendTorobot(move):
    msg4robot = ','.join([move, '250,0,0,0'])
    print("Sending:", msg4robot)
    bytesToSend = str.encode(msg4robot)
    bufferSize = 1024
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPClientSocket.sendto(bytesToSend, robotAddressPort)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    turn = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for landmark in hand_landmarks.landmark:
                h, w, _ = image.shape
                landmarks.append((int(landmark.x * w), int(landmark.y * h)))

            # rotation angle
            wrist = landmarks[0]
            middle_mcp = landmarks[9]
            angle = math.atan2(wrist[1] - middle_mcp[1], wrist[0] - middle_mcp[0])
            angle = math.degrees(angle) - 90
            if angle <= -180:
                angle += 360
            elif angle > 180:
                angle -= 360

            cv2.putText(frame, f"Rotation Angle: {angle:.2f}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # finger fold detection (tip below PIP means folded)
            def is_folded(tip, pip):
                return landmarks[tip][1] > landmarks[pip][1]

            def is_raised(tip, pip):
                return landmarks[tip][1] < landmarks[pip][1]

            index_folded = is_folded(8, 6)
            middle_folded = is_folded(12, 10)
            ring_folded = is_folded(16, 14)
            pinky_folded = is_folded(20, 18)
            thumb_folded = is_folded(4, 2)

            # all fingers folded
            if index_folded and middle_folded and ring_folded and pinky_folded:
                cv2.putText(frame, "Back", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                sendTorobot("b")
                print("Back")

            # backward: thumb folded, all other fingers up
            # elif thumb_folded and \
            #      is_raised(8, 6) and is_raised(12, 10) and is_raised(16, 14) and is_raised(20, 18):
            #     cv2.putText(frame, "Back", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            #                 (255, 255, 255), 2)
            #     sendTorobot("b")
            #     print("Back")

            elif angle > 20:
                cv2.putText(frame, "Right", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                turn = True
                sendTorobot("r")
            elif angle < -20:
                cv2.putText(frame, "Left", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                turn = True
                sendTorobot("l")
            elif not turn:
                cv2.putText(frame, "Forward", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                sendTorobot("f")
                print("Forward")

    else:
        cv2.putText(frame, "Stop", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        sendTorobot("s")
        print("Stop")

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()