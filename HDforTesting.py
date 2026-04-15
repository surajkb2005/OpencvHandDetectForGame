import cv2 as cv
import mediapipe as mp
import time

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

video = cv.VideoCapture(0)

time.sleep(2.0)

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
                lmlist = []

                for handlm, handness in zip(
                    result.multi_hand_landmarks, result.multi_handedness
                ):
                    hand_label = handness.classification[0].label
                    h, w, c = image.shape

                    if hand_label == "Left":
                        lhand = -10
                    else:
                        lhand = 10

                    for id, lm in enumerate(handlm.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmlist.append([lhand, id, cx, cy])
                        cv.circle(image, (cx, cy), 5, (255, 0, 0), cv.FILLED)

                    print(lmlist, "\n")

            cv.imshow("Hand Detection", image)

            if cv.waitKey(1) & 0xFF == 27:
                break

except Exception as e:
    print("Error:", e)

finally:
    print("Releasing video and closing windows.")
    video.release()
    cv.destroyAllWindows()
