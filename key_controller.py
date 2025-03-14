import time
import threading
import pyautogui

class KeyController:
    def __init__(self):
        self.held_keys = set()
        self.lock = threading.Lock()
        
    def press_key(self, key):
        """Press and release a key"""
        try:
            # Convert key to lowercase for consistency
            key = key.lower()
            
            # Handle special keys
            if len(key) > 1:
                if key == "space":
                    pyautogui.press('space')
                elif key == "up":
                    pyautogui.press('up')
                elif key == "down":
                    pyautogui.press('down')
                elif key == "left":
                    pyautogui.press('left')
                elif key == "right":
                    pyautogui.press('right')
                elif key == "enter":
                    pyautogui.press('enter')
                elif key == "esc":
                    pyautogui.press('esc')
                elif key == "tab":
                    pyautogui.press('tab')
                elif key.startswith("f") and key[1:].isdigit():
                    pyautogui.press(key)
                else:
                    pyautogui.press(key)
            else:
                pyautogui.press(key)
        except Exception as e:
            print(f"Error pressing key {key}: {e}")
            
    def hold_key(self, key):
        """Hold a key down until released"""
        try:
            with self.lock:
                key = key.lower()
                
                # Only press if not already held
                if key not in self.held_keys:
                    # Handle special keys like the press_key method
                    if len(key) > 1:
                        if key == "space":
                            pyautogui.keyDown('space')
                        elif key == "up":
                            pyautogui.keyDown('up')
                        elif key == "down":
                            pyautogui.keyDown('down')
                        elif key == "left":
                            pyautogui.keyDown('left')
                        elif key == "right":
                            pyautogui.keyDown('right')
                        elif key == "enter":
                            pyautogui.keyDown('enter')
                        elif key == "esc":
                            pyautogui.keyDown('esc')
                        elif key == "tab":
                            pyautogui.keyDown('tab')
                        elif key.startswith("f") and key[1:].isdigit():
                            pyautogui.keyDown(key)
                        else:
                            pyautogui.keyDown(key)
                    else:
                        pyautogui.keyDown(key)
                    
                    self.held_keys.add(key)
        except Exception as e:
            print(f"Error holding key {key}: {e}")
            
    def release_key(self, key):
        """Release a specific held key"""
        try:
            with self.lock:
                key = key.lower()
                
                if key in self.held_keys:
                    # Handle special keys 
                    if len(key) > 1:
                        if key == "space":
                            pyautogui.keyUp('space')
                        elif key == "up":
                            pyautogui.keyUp('up')
                        elif key == "down":
                            pyautogui.keyUp('down')
                        elif key == "left":
                            pyautogui.keyUp('left')
                        elif key == "right":
                            pyautogui.keyUp('right')
                        elif key == "enter":
                            pyautogui.keyUp('enter')
                        elif key == "esc":
                            pyautogui.keyUp('esc')
                        elif key == "tab":
                            pyautogui.keyUp('tab')
                        elif key.startswith("f") and key[1:].isdigit():
                            pyautogui.keyUp(key)
                        else:
                            pyautogui.keyUp(key)
                    else:
                        pyautogui.keyUp(key)
                    
                    self.held_keys.remove(key)
        except Exception as e:
            print(f"Error releasing key {key}: {e}")
    
    def release_all_keys(self):
        """Release all held keys"""
        with self.lock:
            for key in list(self.held_keys):
                self.release_key(key)
            self.held_keys.clear()