from enum import Enum
from typing import List

import util
from features import Features

IMG_WIDTH = util.IMAGE_SIZE[0]
IMG_HEIGHT = util.IMAGE_SIZE[1]

# Feature stuff
SPIRAL_DIST_MIN = int(IMG_WIDTH * 0.0782) # ~60 at IMAGE_WIDTH = 768
SPIRAL_DIST_MAX = int(IMG_WIDTH * 0.202) # ~155 at IMAGE_WIDTH = 768
SPIRAL_AREA_MIN = int(IMG_WIDTH * IMG_HEIGHT * 0.015) # ~5500 at IMAGE_SIZE = (768, 480)
SPRIAL_AREA_RATIO = 0.5

KEYHOLE_AREA_MIN = int(IMG_WIDTH * IMG_HEIGHT * 0.0542) # ~20000 at IMAGE_SIZE = (768, 480)
KEYHOLE_AREA_MAX = int(IMG_WIDTH * IMG_HEIGHT * 0.0217) # ~8000 at IMAGE_SIZE = (768, 480)
KEYHOLE_DIST_MIN = int(IMG_WIDTH * 0.0586) # ~45 at IMAGE_WIDTH = 768

# Move stuff
SPIRAL_MOVES = int(util.TARGET_FPS * 0.64) # 19 at TARGET_FPS = 30
KEYHOLE_MOVES = int(util.TARGET_FPS * 0.44) # 13 at TARGET_FPS = 30

class Pattern(Enum):
    DEFAULT = 0
    SPIRAL_R = 1
    SPIRAL_L = 2
    KEYHOLE_L = 3
    KEYHOLE_R = 4

def is_spiral(feats: Features):
    """
    Determine whether the given features form a spiral pattern.
    """
    # If the cursor isn't close enough to a potential spiral, return None
    closest_distances = [dists[0] for dists in feats.distances]
    if min(closest_distances) > SPIRAL_DIST_MIN or max(closest_distances) > SPIRAL_DIST_MAX:
        return None

    # Find the two openings of the spiral
    min_dist_group = []
    max_dist_group = []
    for sector, dists in enumerate(feats.distances):
        prev_sector = sector - 1 if sector > 0 else 5
        prev_dist = feats.distances[prev_sector][0]
        diff = dists[0] - prev_dist
        if abs(diff) < 20:
            min_dist_group.append(sector)
        elif diff < 0:
            min_dist_group.append(sector)
        else:
            max_dist_group.append(sector)

    # Ensure there are four obstacles same distance apart
    # and two obstacles slightly farther apart
    if len(min_dist_group) != 4 or len(max_dist_group) != 2:
        return None

    # Ensure connections between obstacles in the sectors are correct
    for sector, sector_conns in enumerate(feats.connections):
        prev_sector = sector - 1 if sector > 0 else 5
        prev_connections = feats.connections[prev_sector][0].tolist()
        conns = sector_conns[0].tolist()
        if prev_connections == [0, 0] and conns != [0, 1]:
            return None

        if prev_connections == [0, 1] and conns != [1, 0]:
            return None

        if prev_connections == [1, 0] and conns != [0, 0]:
            return None

    # Figure out whether the spiral is turning left or right
    guesses = set()
    for sector in max_dist_group:
        prev_sector = sector - 1 if sector > 0 else 5
        next_sector = sector + 1 if sector < 5 else 0

        conditions = []
        for area in (feats.areas[prev_sector][0], feats.areas[next_sector][0]):
            if area > SPIRAL_AREA_MIN:
                conditions.append(1)
            if area < SPIRAL_AREA_MIN * SPRIAL_AREA_RATIO:
                conditions.append(2)

        if conditions == [1, 2]:
            guesses.add(Pattern.SPIRAL_R)
        elif conditions == [2, 1]:
            guesses.add(Pattern.SPIRAL_L)

    return guesses.pop() if len(guesses) == 1 else None

def is_keyhole(feats: Features):
    """
    Determine whether the given features form a keyole pattern.
    """
    # If no obstacles are large enough, return None immediately
    large_obstacles = list(filter(lambda x: x[1][0] > KEYHOLE_AREA_MIN, enumerate(feats.areas)))

    if large_obstacles == [] or len(large_obstacles) > 1:
        return None

    keyhole_sector = large_obstacles[0][0]

    # If the cursor isn't close enough to a potential keyhole, return None
    keyhole_dist = min(feats.distances[keyhole_sector])
    if keyhole_dist > KEYHOLE_DIST_MIN:
        return None

    opposite = keyhole_sector - 3 if keyhole_sector >= 3 else keyhole_sector + 3
    if feats.areas[opposite][0] > KEYHOLE_AREA_MAX:
        return None

    # Find opening that the keyhole is adjacent to
    opening_sector = max((x for x in enumerate(feats.distances) if x[0] != keyhole_sector), key=lambda x: min(x[1]))[0]

    if opening_sector == 5 and keyhole_sector == 0 or keyhole_sector - opening_sector > 0:
        return Pattern.KEYHOLE_R

    if opening_sector == 0 and keyhole_sector == 5 or keyhole_sector - opening_sector < 0:
        return Pattern.KEYHOLE_L

    return None

def determine_pattern(feats: Features):
    """
    Determine what pattern the given features form, if any.
    """
    if (spiral := is_spiral(feats)):
        return spiral

    if (keyhole := is_keyhole(feats)):
        return keyhole

    return Pattern.DEFAULT

def get_move_sequence(pattern: Pattern) -> List[int]:
    """
    Get a list of key presses that would get us through the given pattern.
    """
    if pattern is Pattern.KEYHOLE_L:
        return ([util.KEY_LEFT] * KEYHOLE_MOVES) + ([util.KEY_RIGHT] * KEYHOLE_MOVES)

    if pattern is Pattern.KEYHOLE_R:
        return ([util.KEY_LEFT] * KEYHOLE_MOVES) + ([util.KEY_RIGHT] * KEYHOLE_MOVES)

    if pattern is Pattern.SPIRAL_L:
        return [util.KEY_LEFT] * SPIRAL_MOVES

    if pattern is Pattern.SPIRAL_R:
        return [util.KEY_RIGHT] * SPIRAL_MOVES

    return []
