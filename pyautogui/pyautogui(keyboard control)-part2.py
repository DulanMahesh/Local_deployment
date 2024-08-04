import time
import pyautogui

# Give a moment (half a second) to bring up the application window if needed.
time.sleep(0.5)

# If on a mac OSX machine, use command key instead of ctrl.
hotkey = 'ctrl'

# Open a new tab using a shortcut key.
pyautogui.hotkey(hotkey, 't')

# Give time for the browser to open the tab and be ready for user (typing) input.
time.sleep(1.0)

# Now type a url at a speedy 100 words per minute!
pyautogui.write('http://www.gmail.com', 0.01)

# Bring 'focus' to the URL bar (shortcut key may vary depending on your browser).
time.sleep(0.1)
pyautogui.hotkey(hotkey, 'l')

pyautogui.press('enter')
