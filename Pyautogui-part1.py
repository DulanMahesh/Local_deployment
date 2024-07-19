# drawing using pyautogui on the paint application
# Wait 3 seconds, to switch to the drawing application.
import time
import pyautogui
time.sleep(3.0)

distance = 200
while distance > 0:
        pyautogui.drag(distance, 0, button='left', duration=0.5)   # move right
        distance -= 50
        pyautogui.drag(0, distance, button='left', duration=0.5)   # move down
        pyautogui.drag(-distance, 0, button='left', duration=0.5)  # move left
        distance -= 50
        pyautogui.drag(0, -distance, button='left', duration=0.5)  # move up