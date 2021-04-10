# import pyautogui as gui
import cv2
import numpy as np


cap = cv2.VideoCapture(1)
while 1:
    ret, img = cap.read()
    img = cv2.flip(img, flipCode=1)
    cv2.imshow("test", img)

    if cv2.waitKey(30) == 27:
        break

cv2.destroyWindow("test")
# gui.press('space')


