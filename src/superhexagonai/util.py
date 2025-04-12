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
    
    # Mask out the highscore text (top left)
    t1_x = int(w * 0.28)
    t1_y = int(h * 0.041)
    cv2.rectangle(img, (0, 0), (t1_x, t1_y), (0, 0, 0), thickness=-1)

    left = 0
    p_x = int(w * 0.803)
    p_y = int(h * 0.07)
    color = img[p_y, p_x]
    if len(img.shape) == 3:
        color = color[0]

    if color < 10:
        left = w * 0.04

    # Mask out the 'Hyper Mode' text (top right)
    t2_x_1 = int(w * 0.60 - left)
    t2_x_2 = int(w * 0.82 - left)
    t2_y = int(h * 0.041)
    cv2.rectangle(img, (t2_x_1, 0), (t2_x_2, t2_y), (0, 0, 0), thickness=-1)

    # Mask out the time survived text (top right)
    t3_x = int(w * 0.18 + left)
    t3_y = int(h * 0.09)
    cv2.rectangle(img, (w - t3_x, 0), (w - 1, t3_y), (0, 0, 0), thickness=-1)

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

        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2GRAY)

        if (cv2_img.shape[1], cv2_img.shape[0]) != final_size:
            cv2_img = cv2.resize(cv2_img, final_size)
        mask_out_score(cv2_img)

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

    return cv2_img
