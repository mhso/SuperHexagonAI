from time import sleep, time

import cv2

import util

# Run 30 times a second.
TIME_TO_SLEEP = 1 / util.TARGET_FPS

def game_loop():
    while True:
        time_start = time()
        screen_grab = util.grab_screenshot(util.IMAGE_SIZE)

        if util.is_game_over(screen_grab):
            break

        yield screen_grab

        frame_duration = time() - time_start + 0.001

        if frame_duration < TIME_TO_SLEEP:
            sleep(TIME_TO_SLEEP - frame_duration)
        elif frame_duration > TIME_TO_SLEEP:
            print(f"Frame duration was: {frame_duration:.3f}s, Expected: {TIME_TO_SLEEP:.3f}s")

        cv2.waitKey(1)
