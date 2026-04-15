import cv2 as cv
import mediapipe as mp
import time

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

video = cv.VideoCapture(0)

time.sleep(2.0)

try:
    while True:
        ret, image = video.read()

        if not ret:
            break
        
        image = cv.flip(image,1)

        cv.imshow("Hand Detection",image)

        if cv.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print("Error:", e)

finally:
    video.release()
    cv.destroyAllWindows()