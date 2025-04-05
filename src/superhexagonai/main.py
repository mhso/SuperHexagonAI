from time import sleep

from pynput.keyboard import Controller

import util
import heuristic as heuristic
import features
import patterns
from game_loop import game_loop

KEYBOARD = Controller()
PATTERN_GAP = util.TARGET_FPS // 10

def run():
    """
    Run the self-playing bot. Runs in a loop, until 'q' is pressed
    or the program is aborted.
    """
    while True:
        iteration = 0
        pattern = None
        prev_pattern_iter = 0
        held_key = None
        feats: features.Features = None
        moves = []
        for image in game_loop():
            if not moves and (new_feats := features.analyze_img(image)) is not None:
                feats = new_feats

            if feats is not None:
                new_pattern = patterns.determine_pattern(feats)
                if pattern is None or iteration - prev_pattern_iter > PATTERN_GAP:
                    pattern = new_pattern
                    moves = patterns.get_move_sequence(pattern)
                    if pattern is not patterns.Pattern.DEFAULT:
                        prev_pattern_iter = iteration + len(moves)

                if not moves:
                    scores, layers = heuristic.get_scores_and_layers(feats)
                    moves = [heuristic.get_move(scores, layers, feats.cursor)]

            if moves:
                key = moves.pop(0)

                if held_key is not None and key != held_key:
                    KEYBOARD.release(held_key)

                if key != 0:
                    key = util.KEY_MAP[key-1]
                    util.press_key(KEYBOARD, key, held_key)
                    held_key = key

            elif held_key is not None:
                KEYBOARD.release(held_key)

            iteration += 1

        print("We died! Press 'Space' to go again or 'q' to quit...")

        # Game over! Wait for new round.
        while not util.key_pressed("space"):
            if util.key_pressed("q"):
                exit(0)

            sleep(0.05)

        print("Restarting...")

        sleep(1)

if __name__ == "__main__":
    print("Press 'Space' at the 'Hexagonest' level screen to start playing...")

    while not util.key_pressed("space"):
        sleep(0.01)

    print("Now playing...")

    sleep(0.5)

    run()
