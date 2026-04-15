import cv2 as cv
import mediapipe as mp
import time

# importing arduino code to controll led's
from Arduino import controllers as cnt

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

video = cv.VideoCapture(0)

time.sleep(2.0)

tipids = [4, 8, 12, 16, 20]


def getfingers(lmlist_l, lmlist_r):
    fingers = []

    # if len(lmlist_l) != 0:
    #     for id in range(1, 5):
    #         if lmlist_l[tipids[id]][2] < lmlist_l[tipids[id] - 2][2]:
    #             fingers.append(1)
    #         else:
    #             fingers.append(0)
    #     if lmlist_l[tipids[0]][1] > lmlist_l[tipids[0] - 1][1]:
    #         fingers.append(1)
    #     else:
    #         fingers.append(0)
    # else:
    #     for id in range(0, 5):
    #         fingers.append(0)

    if len(lmlist_r) != 0:
        if lmlist_r[tipids[0]][1] < lmlist_r[tipids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lmlist_r[tipids[id]][2] < lmlist_r[tipids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
    else:
        for id in range(0, 5):
            fingers.append(0)

    return fingers


def getlmlist(handlm, image):
    lmlist = []
    h, w, shape = image.shape
    for id, lm in enumerate(handlm.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmlist.append([id, cx, cy])
        image = cv.circle(image, (cx, cy), 5, (255, 0, 255), cv.FILLED)
    return lmlist


try:
    with mp_hand.Hands(
        max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, image = video.read()

            if not ret:
                print("Failed to grab frame")
                break

            image = cv.flip(image, 1)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            result = hands.process(image)
            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            if not result.multi_hand_landmarks:
                cv.putText(
                    image,
                    "No hands detected",
                    (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv.imshow("Hand Detection", image)

                if cv.waitKey(1) & 0xFF == 27:
                    break
                continue

            if result.multi_hand_landmarks:
                lmlist_l = []
                lmlist_r = []

                for handlm, handness in zip(
                    result.multi_hand_landmarks, result.multi_handedness
                ):
                    hand_label = handness.classification[0].label
                    h, w, c = image.shape

                    if hand_label == "Left":
                        lmlist_l = getlmlist(handlm, image)
                    else:
                        lmlist_r = getlmlist(handlm, image)

                    fingers = getfingers(lmlist_l, lmlist_r)

                    print(fingers)

                    # send data to arduino
                    cnt.led(fingers)
                    time.sleep(0.1)

            cv.imshow("Hand Detection", image)

            if cv.waitKey(1) & 0xFF == 27:
                break

except Exception as e:
    print("Error:", e)

finally:
    print("Releasing video and closing windows.")
    video.release()
    cv.destroyAllWindows()
