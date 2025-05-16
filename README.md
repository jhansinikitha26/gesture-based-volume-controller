# gesture-based-volume-controller

import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not found or cannot be accessed.")
    exit()

print("Step 1: Webcam initialized.")

# Initialize MediaPipe Hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
print("Step 2: MediaPipe hand tracker initialized.")

# Initialize pycaw for audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]
print(f"Step 3: Volume range is {min_vol} dB to {max_vol} dB")

while True:
    # Capture frame-by-frame
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image.")
        break

    # Convert the BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Process detected hands
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw hand landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Extract landmarks for thumb and index finger tips
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            if lmList:
                # Coordinates of thumb and index finger tips
                x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
                x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip

                # Calculate the distance between the two fingertips
                length = math.hypot(x2 - x1, y2 - y1)
                print(f"Distance between fingers: {length:.2f} pixels")

                # Map the length to the volume range
                vol = np.interp(length, [30, 200], [min_vol, max_vol])
                volume.SetMasterVolumeLevel(vol, None)
                print(f"Volume set to {vol:.2f} dB")

                # Draw circles at the fingertips and a line between them
                cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # Display volume bar
                vol_bar = np.interp(length, [30, 200], [400, 150])
                cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
                cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 255, 0), cv2.FILLED)

                # Display volume percentage
                vol_perc = np.interp(length, [30, 200], [0, 100])
                cv2.putText(img, f'{int(vol_perc)}%', (40, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    else:
        print("No hand detected.")

    # Display the resulting frame
    cv2.imshow("Volume Gesture Control", img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program...")
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("Webcam and windows closed.")


 
 






 




 










 
