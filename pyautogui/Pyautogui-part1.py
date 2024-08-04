#drawing using pyautogui on the paint application
# Import the required libraries
import time
import pyautogui

# Wait for 3 seconds to give you time to switch to the drawing application
time.sleep(3.0)

# Initialize the distance for the first move
distance = 600

# Loop to create the spiral pattern
while distance > 0:
    # Move the mouse right by 'distance' pixels while holding down the left mouse button
    pyautogui.drag(distance, 0, button='left', duration=0.5)   # move right
    # Decrease the distance by 50 pixels
    distance -= 50
    # Move the mouse down by 'distance' pixels while holding down the left mouse button
    pyautogui.drag(0, distance, button='left', duration=0.5)   # move down
    # Move the mouse left by 'distance' pixels while holding down the left mouse button
    pyautogui.drag(-distance, 0, button='left', duration=0.5)  # move left
    # Decrease the distance by 50 pixels
    distance -= 50
    # Move the mouse up by 'distance' pixels while holding down the left mouse button
    pyautogui.drag(0, -distance, button='left', duration=0.5)  # move up
