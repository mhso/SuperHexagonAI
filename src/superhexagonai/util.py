from PIL import Image
import win32gui
import win32ui
import win32api
import win32con
import numpy as np
import cv2
from cv2.typing import MatLike
from pynput.keyboard import Key

IMAGE_SIZE = (768, 480)
TARGET_FPS = 30

KEY_HEX_CODES = {
    "left": 0x25, "right": 0x27, "space": 0x20, "q": 0x51
}

KEY_STATES = {
    "left": win32api.GetKeyState(KEY_HEX_CODES["left"]),
    "right": win32api.GetKeyState(KEY_HEX_CODES["right"]),
    "space": win32api.GetKeyState(KEY_HEX_CODES["space"]),
    "q": win32api.GetKeyState(KEY_HEX_CODES["q"])
}

KEY_LEFT = 1
KEY_RIGHT = 2

KEY_MAP = [Key.left, Key.right, Key.space]

def press_key(keyboard, key, held_key=None):
    if held_key is not None:
        keyboard.release(held_key)
    keyboard.press(key)

def key_pressed(key: str) -> bool:
    new_key_state = win32api.GetKeyState(KEY_HEX_CODES[key])
    old_key_state = KEY_STATES[key]
    return new_key_state != old_key_state and new_key_state < 0

def is_game_over(img: MatLike) -> bool:
    """
    Determine whether the given image indicates that we have died in the game.
    This is case if it's a completely blank white image.
    """
    total_pixels = 1
    for dim in img.shape:
        total_pixels *= dim

    return np.count_nonzero(img == 255) > total_pixels * 0.75

def get_super_hexagon_window() -> int:
    """
    Get the win32gui handle associated with the 'Super Hexagon' game window,
    if the game is running. Returns None otherwise.
    """
    hwnd = win32gui.FindWindow(None, "Super Hexagon")
    if not win32gui.IsWindow(hwnd):
        return None
    return hwnd

def mask_out_score(img: MatLike):
    """
    Mask out highscore and time survived text at the top of the image.
    """
    h, w = img.shape[:2]

    # Mask out the 'Hyper Mode' text (top right)
    x_1 = int(w * 0.6)
    x_2 = int(w * 0.836)
    y = int(h * 0.06)
    cv2.rectangle(img, (x_1, 0), (x_2, y), (0, 0, 0), thickness=-1)

    # Mask out the time survived text (top right)
    x = int(w * 0.18)
    y = int(h * 0.1)
    cv2.rectangle(img, (w - x, 0), (w - 1, y), (0, 0, 0), thickness=-1)

    # Mask out the highscore text (top left)
    x = int(w * 0.25)
    cv2.rectangle(img, (0, 0), (x, y), (0, 0, 0), thickness=-1)

def grab_screenshot(final_size: tuple[int, int]) -> MatLike:
    """
    Take a screenshot of Super Hexagon using various pywin32 calls.
    These are platform specific (unfortunately), but are a lot faster than
    using Pillow or similar.
    """
    try:
        hwnd = get_super_hexagon_window()
        win32gui.SetForegroundWindow(hwnd)

        x_1, y_1, x_2, y_2 = win32gui.GetWindowRect(hwnd)
        offset_y = 30
        offset_x = 10
        sc_w = (x_2 - x_1) - (offset_x * 2)
        sc_h = int((y_2 - y_1) - (offset_y * 1.5))

        window_dc = win32gui.GetWindowDC(hwnd)
        dc_obj = win32ui.CreateDCFromHandle(window_dc)

        compat_dc = dc_obj.CreateCompatibleDC()
        data_bitmap = win32ui.CreateBitmap()
        data_bitmap.CreateCompatibleBitmap(dc_obj, sc_w, sc_h)
        compat_dc.SelectObject(data_bitmap)
        compat_dc.BitBlt((0, 0), (sc_w, sc_h), dc_obj, (offset_x, offset_y), win32con.SRCCOPY)
        bitmap_bytes = data_bitmap.GetBitmapBits(True)
        img = Image.frombuffer("RGB", (sc_w, sc_h), bitmap_bytes, "raw", "BGRX", 0, 1)

        img_arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2GRAY)

        resized = cv2.resize(img_arr, final_size)
        mask_out_score(resized)

    finally:
        # Free resources
        if dc_obj:
            dc_obj.DeleteDC()
        if compat_dc:
           compat_dc.DeleteDC()
        if window_dc:
            win32gui.ReleaseDC(hwnd, window_dc)
        if data_bitmap:
           win32gui.DeleteObject(data_bitmap.GetHandle())

    return resized
