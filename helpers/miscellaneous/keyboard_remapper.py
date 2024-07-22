from pynput.keyboard import Key, Listener, Controller
keyboard = Controller()

def on_press(key):
    print(key)
    if key == Key.tab:
        keyboard.press('q')
        keyboard.release('q')
    if key == Key.enter:
        keyboard.press('w')
        keyboard.release('w')

with Listener(on_press=on_press) as listener:
    listener.join()