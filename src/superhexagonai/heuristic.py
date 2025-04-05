import numpy as np

from features import MAX_DIST, Features
import util

TRAVEL_PENALTY = 35

def get_distance_to_sector(sector, cursor):
    """
    Get a value for the distance between the sector the cursor is in
    and the target sector.
    """
    clockwise = sector < 3 or sector == 3 and cursor < 0

    if clockwise:
        return sector + 1 - cursor

    return 6 - sector + cursor

def calculate_risk_score(sector, wall_dist, cursor):
    """
    Get the score for the given sector, distance, and cursor.
    """
    travel_distance = get_distance_to_sector(sector, cursor)

    return (MAX_DIST - wall_dist) + (travel_distance * TRAVEL_PENALTY)

def adjust_opposite_sector_scores(scores: np.ndarray, cursor: float):
    """
    Adjust scores of sectors on the opposite side of where the cursor is,
    based on the weights of in-between sectors. This disuades risky moves that
    cut across sectors where walls are very close.
    """
    sectors_to_adjust = [3, 2, 4]

    for sector in sectors_to_adjust:
        if sector == 2:
            in_between = [scores[1][0]]
        elif sector == 3:
            if cursor < 0:
                in_between = [scores[4][0], scores[5][0]]
            else:
                in_between = [scores[1][0], scores[2][0]]
        elif sector == 4:
            in_between = [scores[5][0]]

        for blocking_sector_score in in_between:
            mult = 0.5 if blocking_sector_score > MAX_DIST * 0.75 else 0.25
            scores[sector, 0] += (blocking_sector_score * mult)

    return scores

def group_distances_by_layer(feats: Features):
    """
    Group distances to walls in each sector by their connections to form
    layers. This is used to determine whether the current layer only has one exit.
    """
    dense_layers = []
    for layer_index in range(6):
        walls_in_layer = [None] * 6
        for sector, (connections, distances) in enumerate(zip(feats.connections, feats.distances)):
            if connections[layer_index].tolist() != [0, 0]:
                walls_in_layer[sector] = distances[layer_index]

        dense_layers.append(walls_in_layer)

    return dense_layers

def get_scores_and_layers(feats: Features):
    """
    Calculate risk scores for each wall in each sector of the hexagon.
    Higher values mean going to that sector is riskier.
    """
    scores = [[] for _ in range(6)]
    for sector, dist_list in enumerate(feats.distances):
        for dist in dist_list:
            scores[sector].append(calculate_risk_score(sector, dist, feats.cursor))

    scores = adjust_opposite_sector_scores(np.array(scores), feats.cursor)

    layers = group_distances_by_layer(feats)

    return scores, layers or None

def get_only_exit(layers: np.ndarray) -> bool:
    """
    Goes through the first layer of the obstacles and returns the sector
    where the only exit is, if such an exit exists. Otherwise, returns None.
    """
    exits_in_layer = []

    if layers is not None:
        for sector, distance in enumerate(layers[0]):
            if distance is None:
                exits_in_layer.append(sector)

    return exits_in_layer[0] if len(exits_in_layer) == 1 else None

def get_move(scores: np.ndarray, layers: np.ndarray, cursor: float) -> int:
    """
    Get the key that should be pressed based on the given scores, layers,
    and relative position of the cursor. Return value is one of:
    
    0: Press nothing (keep going straight)
    1: Press left arrow
    2: Press right arrow
    """
    # If there is only one exit in the current layer, go to it.
    if (exit_sector := get_only_exit(layers)):
        target_sector = exit_sector

    # Go to the sector with the lowest score
    else:
        target_sector = 0
        min_score = 1000
        for sector, scores_in_sector in enumerate(scores):
            if scores_in_sector[0] < min_score:
                target_sector = sector
                min_score = scores_in_sector[0]

    # We are already in the desired sector
    if target_sector == 0:
        return 0

    # If we should go counter-clockwise, move left
    if target_sector > 3 or target_sector == 3 and cursor < 0:
        return util.KEY_LEFT

    # Otherwise, move right
    return util.KEY_RIGHT
