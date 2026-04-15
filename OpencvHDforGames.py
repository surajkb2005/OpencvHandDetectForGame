import cv2
import mediapipe as mp

# for controlling the keyboard
import time
from directkeys import left_arrow, right_arrow
from directkeys import PressKey, ReleaseKey

left_key_pressed = left_arrow
right_key_pressed = right_arrow


time.sleep(2.0)
current_key_pressed = set()

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
video = cv2.VideoCapture(0)

tipIds = [4, 8, 12, 16, 20]
try:
    with mp_hand.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while True:
            keyPressed = False
            left_active = False
            right_active = False
            key_count = 0
            key_pressed = 0

            ret, image = video.read()
            if not ret:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # Makes the image read-only to improve performance while MediaPipe processes it (no need to modify pixels).
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            lmlist = []
            if results.multi_hand_landmarks:
                hand_landmark = results.multi_hand_landmarks[0]
                h, w, c = image.shape
                for id, lm in enumerate(hand_landmark.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmlist.append([id, cx, cy])
                mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)

            fingers = []
            if len(lmlist) != 0:
                if lmlist[tipIds[0]][1] > lmlist[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                # because i need tip id from 1 to 4 so thumb is not included in this loop
                for id in range(1, 5):
                    # In lmlist, 1 is x axis and 2 is y axis
                    if lmlist[tipIds[id]][2] < lmlist[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                total = fingers.count(
                    1
                )  # counts the number of fingers that are up (ie. 1)
                if total == 0:
                    cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
                    cv2.putText(
                        image,
                        "Left",
                        (45, 375),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 0, 0),
                        5,
                    )
                    if left_key_pressed not in current_key_pressed:
                        PressKey(left_key_pressed)
                    left_active = True
                    current_key_pressed.add(left_key_pressed)
                    key_pressed = left_key_pressed
                    keyPressed = True
                    key_count = key_count + 1
                elif total == 5:
                    cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
                    cv2.putText(
                        image,
                        "Right",
                        (45, 375),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (255, 0, 0),
                        5,
                    )
                    if right_key_pressed not in current_key_pressed:
                        PressKey(right_key_pressed)
                    key_pressed = right_key_pressed
                    right_active = True
                    keyPressed = True
                    current_key_pressed.add(right_key_pressed)
                    key_count = key_count + 1

            if not keyPressed:
                for key in current_key_pressed:
                    ReleaseKey(key)
                current_key_pressed = set()

            else:
                for key in list(current_key_pressed):
                    if key != key_pressed:
                        ReleaseKey(key)
                        current_key_pressed.remove(key)

            cv2.imshow("Frame", image)
            k = cv2.waitKey(1)
            if k & 0xFF == 27:
                break
            if k & 0xFF == ord("q"):
                break

except KeyboardInterrupt:
    print("Program interrupted by user")
except Exception as e:
    print("Program failed with error:", e)
finally:
    print("Cleaning up resources...")
    try:
        for key in current_key_pressed:
            ReleaseKey(key)
    except:
        pass  # avoids crash if variable doesn't exist

    if video.isOpened():
        video.release()

    cv2.destroyAllWindows()
