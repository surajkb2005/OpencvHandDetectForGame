import cv2
import mediapipe as mp
import time

# for controlling the keyboard
from directkeys import right_pressed, left_pressed, up_pressed, down_pressed
from directkeys import PressKey, ReleaseKey

left_key_pressed = left_pressed
right_key_pressed = right_pressed
up_key_pressed = up_pressed
down_key_pressed = down_pressed

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
video = cv2.VideoCapture(0)

current_key_pressed = set()
time.sleep(2.0)

tipIds = [4, 8, 12, 16, 20]


def count_fingers(lmlist, hand_label):
    fingers = []

    # Thumb (different logic for left/right)
    if hand_label == "Right":
        fingers.append(1 if lmlist[tipIds[0]][1] > lmlist[tipIds[0] - 1][1] else 0)
    else:
        fingers.append(1 if lmlist[tipIds[0]][1] < lmlist[tipIds[0] - 1][1] else 0)

    # Other fingers
    for i in range(1, 5):
        fingers.append(1 if lmlist[tipIds[i]][2] < lmlist[tipIds[i] - 2][2] else 0)

    return fingers.count(1)


def update_keys(required_keys):
    global current_key_pressed

    # Press new keys
    for key in required_keys:
        if key not in current_key_pressed:
            PressKey(key)

    # Release unused keys
    for key in list(current_key_pressed):
        if key not in required_keys:
            ReleaseKey(key)

    current_key_pressed = set(required_keys)


try:
    with mp_hand.Hands(
        max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while True:
            # keyPressed = False
            # left_active = False
            # right_active = False
            # key_count = 0
            # key_pressed = 0

            ret, image = video.read()
            if not ret:
                break

            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # Makes the image read-only to improve performance while MediaPipe processes it (no need to modify pixels).
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            left_state = None  # OPEN / CLOSED
            right_state = None  # OPEN / CLOSED

            if not results.multi_hand_landmarks:
                update_keys([])
                cv2.imshow("Hand Control", image)

                k = cv2.waitKey(1)
                if k & 0xFF == 27:
                    break
                if k & 0xFF == ord("q"):
                    break
                continue
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                ):

                    label = handedness.classification[0].label  # Left or Right

                    h, w, _ = image.shape
                    lmlist = []

                    for id, lm in enumerate(hand_landmarks.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmlist.append([id, cx, cy])

                    mp_draw.draw_landmarks(
                        image, hand_landmarks, mp_hand.HAND_CONNECTIONS
                    )

                    fingers = count_fingers(lmlist, label)

                    if fingers >= 4:
                        state = "OPEN"
                    elif fingers <= 1:
                        state = "CLOSED"
                    else:
                        state = "MID"

                    if label == "Left":
                        left_state = state
                    else:
                        right_state = state

            action = "STOP"
            required_keys = []

            if left_state == "OPEN" and right_state == "OPEN":
                action = "FORWARD"
                required_keys = [up_key_pressed]

            elif left_state == "CLOSED" and right_state == "OPEN":
                action = "LEFT"
                required_keys = [up_key_pressed, left_key_pressed]

            elif left_state == "OPEN" and right_state == "CLOSED":
                action = "RIGHT"
                required_keys = [up_key_pressed, right_key_pressed]

            elif left_state == "CLOSED" and right_state == "CLOSED":
                action = "REVERSE"
                required_keys = [down_key_pressed]

            # one-hand handling
            elif left_state == "OPEN" and right_state is None:
                action = "STOP"
                required_keys = []

            elif right_state == "OPEN" and left_state is None:
                action = "STOP"
                required_keys = []

            # Apply key updates
            update_keys(required_keys)

            # Display
            cv2.putText(
                image,
                f"Action: {action}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Hand Control", image)
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
        current_key_pressed.clear()
    except:
        pass  # avoids crash if variable doesn't exist

    if video.isOpened():
        video.release()

    cv2.destroyAllWindows()
