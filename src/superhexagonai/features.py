import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
from cv2.typing import MatLike, Rect
import numpy as np

from util import IMAGE_SIZE

IMG_WIDTH = IMAGE_SIZE[0]
IMG_HEIGHT = IMAGE_SIZE[1]

MAX_SLOPE = 1_000_000
MAX_DIST = IMG_HEIGHT - 80 # ~400 at IMAGE_HEIGHT = 480
HEXAGON_SPLIT_THICKNESS = 25
DIST_BETWEEN_LAYERS = int(IMG_WIDTH * 0.065) # ~50 at IMAGE_WIDTH = 768

MIN_CONTOUR_AREA = int(IMG_WIDTH * IMG_HEIGHT * 0.0012) # ~400 at IMAGE_SIZE = (768, 480)
MAX_CONTOUR_AREA = int(IMG_WIDTH * IMG_HEIGHT * 0.62) # ~200.000 at IMAGE_SIZE = (768, 480)

CURSOR_MIN_AREA = int(IMG_WIDTH * IMG_HEIGHT * 0.0000775) # ~25 at IMAGE_SIZE = (768, 480)
CURSOR_MAX_AREA = int(IMG_WIDTH * IMG_HEIGHT * 0.00031) # ~100 at IMAGE_SIZE = (768, 480)
CURSOR_DIST_MIN = int(IMG_WIDTH * 0.0455) # ~35 at IMAGE_WIDTH = 768
CURSOR_DIST_MAX = int(IMG_WIDTH * 0.1041) # ~80 at IMAGE_WIDTH = 768

THRESHOLD = 220
BORDER_THICKNESS = 5

@dataclass
class Features:
    """
    Dataclass that contains the relevant features extracted
    from a screenshot of the game.
    """
    cursor: float
    distances: np.ndarray
    connections: np.ndarray
    areas: np.ndarray

def get_threshold_img(img: MatLike, threshold: int, border: bool = False) -> MatLike:
    """
    Create a binary image by performing thresholding on the given input image.
    Border specifies whether a white border will be added to the image.
    """
    copy = img.copy()
    if len(copy.shape) > 2:
        copy = cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY)

    thresh = cv2.threshold(copy, threshold, 255, cv2.THRESH_BINARY_INV)[1]

    if border:
        # Color the borders white
        h, w = thresh.shape[:2]
        thresh[:h, :BORDER_THICKNESS] = 255
        thresh[:h, w-BORDER_THICKNESS:w] = 255
        thresh[:BORDER_THICKNESS, :w] = 255
        thresh[h-BORDER_THICKNESS:, :w] = 255

    return thresh

def get_contours(thresh: MatLike, offset=(0, 0)) -> List[MatLike]:
    """
    Find contours in the given binary image using OpenCV.
    """
    return cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE, offset=offset)[0]

def get_mid_bbox(bbox: Rect) -> Tuple[int, int]:
    """
    Return the mid point of the given rectangle.
    """
    mid_x = bbox[0] + (bbox[2]/2)
    mid_y = bbox[1] + (bbox[3]/2)
    return mid_x, mid_y

def eucl_dist(p_1: Tuple[float, float], p_2: Tuple[float, float]) -> float:
    """
    Calculate the euclidean distance between two points.
    """
    dist_x_sq = (p_1[0] - p_2[0]) ** 2
    dist_y_sq = (p_1[1] - p_2[1]) ** 2
    return math.sqrt(dist_x_sq + dist_y_sq)

def sort_hexagon_points(sides: np.ndarray) -> MatLike:
    """
    Sort the points that make up the center hexagon in clockwise order.
    """
    def sort_x(val):
        return val[0][0]

    def sort_y(val):
        return val[0][1]

    x_sorted = sorted(sides, key=sort_x)
    y_sorted1 = sorted(x_sorted[3:], key=sort_y)
    y_sorted2 = sorted(x_sorted[:3], key=sort_y)
    y_sorted2.reverse()
    return np.array(y_sorted1 + y_sorted2)

def extract_hexagon(img: MatLike) -> MatLike:
    """
    Find and extract the six points that make up the spinning hexagon
    in the middle of the screen.
    """
    grayscale = img.copy()
    if len(grayscale.shape) > 2:
        grayscale = cv2.cvtColor(grayscale, cv2.COLOR_RGB2GRAY)

    height, width = grayscale.shape
    center_x = width // 2
    center_y = height // 2

    mask = np.zeros((height + 2, width + 2), np.uint8)
    flood_filled = grayscale.copy()

    # Do floodfill, to create a single unbroken contour for the hexagon
    cv2.floodFill(flood_filled, mask, (center_x, center_y), 255)

    im_inv = cv2.bitwise_not(flood_filled)
    thresh = get_threshold_img(grayscale | im_inv, THRESHOLD)

    contours = get_contours(thresh)
    epsilon = 0.005 * cv2.arcLength(contours[0], True)
    hexagon = cv2.approxPolyDP(contours[0], epsilon, True)

    # Attempts to construct a polygon fitted to the six sides of the hexagon,
    # so we can use its corners to split the screen into sectors.
    # Attempts to do so six times with progressively lower threshold for precision.
    attempt = 0
    attempts = 6
    while len(hexagon) != attempts and attempt < attempts:
        eps = 0.005 + (0.005 * attempt)
        epsilon = eps * cv2.arcLength(contours[0], True)
        hexagon = cv2.approxPolyDP(contours[0], epsilon, True)
        attempt += 1

    return sort_hexagon_points(hexagon)

def extract_cursor(img: MatLike, mid_p: Tuple[int, int]) -> Optional[MatLike]:
    """
    Find and return the contour that form the cursor.
    """
    contours = get_contours(img)

    cursor = None
    for contour in contours:
        area = cv2.contourArea(contour)
        bbox = cv2.boundingRect(contour)
        mid_cont = get_mid_bbox(bbox)

        dist_to_mid = eucl_dist(mid_p, mid_cont)

        if dist_to_mid < CURSOR_DIST_MIN or dist_to_mid > CURSOR_DIST_MAX or len(contour) < 5:
            continue

        triangle_area = cv2.minEnclosingTriangle(contour)[0]
        if CURSOR_MIN_AREA < area < CURSOR_MAX_AREA and area / triangle_area > 0.7:
            cursor = contour

    return cursor

def invert_y(y):
    return IMG_HEIGHT - y

def get_slope(p_1, p_2):
    x_alpha = p_2[0] - p_1[0]
    y_alpha = p_2[1] - p_1[1]
    return math.atan2(y_alpha, x_alpha)

def calculate_slope(corner: Tuple[int, int], opp_corner: Tuple[int, int]) -> Tuple[List[float], List[float]]:
    """
    Calculates the slope and intercept of the line that
    is formed by two given points of the middle hexagon.
    """
    x_1 = opp_corner[0]
    y_1 = invert_y(opp_corner[1])

    x_2 = corner[0]
    y_2 = invert_y(corner[1])

    x_delta = x_2 - x_1
    if x_delta == 0: # Slope is infinite.
        slope = MAX_SLOPE if y_2 > y_1 else -MAX_SLOPE
        intercept = x_1
    else:
        y_delta = y_2 - y_1
        if y_delta == 0: # Slope is 0.
            intercept = 0
            slope = 0
        else:
            slope = y_delta / x_delta
            intercept = (slope * x_2 - y_2) * -1

    return slope, intercept

def get_hexagon_slopes(hexagon: MatLike) -> Tuple[List[float], List[float]]:
    """
    Get the slopes and intercepts of the lines that form the six sectors
    that obstacles can appear in.
    """
    slopes = [None] * 6
    intercepts = [None] * 6

    for i in range(0, 3):
        slope, intercept = calculate_slope(hexagon[i][0], hexagon[i+3][0])
        slopes[i] = slope
        slopes[i+3] = slope
        intercepts[i] = intercept
        intercepts[i+3] = intercept

    return slopes, intercepts

def get_sector_line_endpoints(hexagon: MatLike, slopes: List[float], intercepts: List[float]) -> List[Tuple[float, float]]:
    """
    Get the points that when paired with each corner of the middle hexagon
    form lines that define the edge of each sector.
    """
    endpoints = [None] * len(hexagon)

    for i, side in enumerate(hexagon):
        x_target = 0
        y_target = 0

        if side[0][0] < IMG_WIDTH/2:
            if slopes[i] < 0:
                y_target = IMG_HEIGHT
            else:
                y_target = 0
        else:
            if slopes[i] < 0:
                y_target = 0
            else:
                y_target = IMG_HEIGHT

        if slopes[i] == 0:
            x_target = 0 if side[0][0] < IMG_WIDTH / 2 else IMG_WIDTH
            y_target = IMG_HEIGHT / 2
        elif abs(slopes[i]) == MAX_SLOPE: # Infinite slope.
            x_target = intercepts[i]
        else:
            x_target = (y_target - intercepts[i]) / slopes[i]

        endpoints[i] = (x_target, y_target)

    return endpoints

def split_screen_into_sectors(img: MatLike, hexagon: MatLike) -> List[Tuple[float, float]]:
    """
    Split the screen into the six sectors where obstacles can appear by drawing
    lines that matches the edges of each sector.
    """
    slopes, intercepts = get_hexagon_slopes(hexagon)
    targets = get_sector_line_endpoints(hexagon, slopes, intercepts)

    for i in range(0, 3):
        target = targets[i]
        target_opp = targets[i+3]
        x1, y1 = target_opp
        x2, y2 = target

        cv2.line(
            img,
            (int(x1), int(invert_y(y1))),
            (int(x2), int(invert_y(y2))),
            255,
            HEXAGON_SPLIT_THICKNESS
        )

    return targets

def get_mid_exclusion_area(hexagon: MatLike, cursor_contour: MatLike) -> Tuple[Tuple[int, int], int]:
    """
    Get a point and radius around the middle hexagon where any contour that
    has a midpoint that lies within this radius is ignored.
    """
    mid_p = get_mid_bbox(cv2.boundingRect(hexagon))
    cursor_p = get_mid_bbox(cv2.boundingRect(cursor_contour))
    return mid_p, eucl_dist(mid_p, cursor_p)

def filter_contour_by_midpoint(contour: MatLike, mid_p: Tuple[int, int], radius: int):
    """
    Return a boolean that indicates whether the midpoint of the given contour likes
    within the circle formed by the given point and radius.
    """
    bbox = cv2.boundingRect(contour)
    x, y = get_mid_bbox(bbox)
    within_mid = (x - mid_p[0]) ** 2 + (y - mid_p[1]) ** 2 < radius ** 2

    return not within_mid

def filter_contour_by_area(contour: MatLike):
    """
    Return a boolean that indicates whether the given contour is too large to be relevant.
    """
    return MIN_CONTOUR_AREA < cv2.contourArea(contour) < MAX_CONTOUR_AREA

def filter_contours(contours: MatLike, hexagon: MatLike, cursor_contour: MatLike, filter_midpoint: bool = True):
    """
    Filter the given contours using the functions `filter_contour_by_area`
    and `filter_contour_by_midpoint`, if `filter_midpoint` is True.
    """
    mid_p, radius = get_mid_exclusion_area(hexagon, cursor_contour)

    def filter_func(c):
        return (
            (filter_contour_by_midpoint(c, mid_p, radius) if filter_midpoint else True)
            and filter_contour_by_area(c)
        )

    return list(filter(filter_func, contours))

def get_contour_connections(all_contours: List[MatLike], sector_contours: List[MatLike]) -> List[List[int]]:
    """
    Find connections between contours across sectors of the hexagon.
    This will help us navigate the obstacles later on.
    """
    connections = [[] for _ in sector_contours]

    # Filter out contours whose midpoint don't lie within the bounding box
    # of a full contour
    full_bboxes = ((cont, cv2.boundingRect(cont)) for cont in all_contours)
    sector_midpoints = [(cont, get_mid_bbox(cv2.boundingRect(cont))) for cont in sector_contours]

    potential_pairs = []
    for full_cont, full_bbox in full_bboxes:
        pairs_for_contour = []
        for index, (sector_cont, sector_mid) in enumerate(sector_midpoints):
            if (
                full_bbox[0] < sector_mid[0] < full_bbox[0] + full_bbox[2]
                and full_bbox[1] < sector_mid[1] < full_bbox[1] + full_bbox[3]
            ):
                pairs_for_contour.append((index, sector_cont))

        potential_pairs.append(pairs_for_contour)

    # Go through remaining contours and find those that are contained within
    # a full contour. Save the connection
    for full_cont, pairs in zip(all_contours, potential_pairs):
        full_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
        cv2.drawContours(full_img, [full_cont], -1, 255, -1)
        count_full = cv2.countNonZero(full_img)
        if count_full == 0:
            continue

        connected_sectors = []
        for index, sector_cont in pairs:
            sector_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
            cv2.drawContours(sector_img, [sector_cont], -1, 255, -1)
            count_sector = cv2.countNonZero(sector_img)
            if count_sector == 0:
                continue

            count_combined = cv2.countNonZero(full_img + sector_img)

            if count_combined - count_full < 10:
                connected_sectors.append(index)

        for index in connected_sectors:
            connections[index].extend([i for i in connected_sectors if i != index])

    return connections

def get_sector_connections(contour_connections: List[List[int]]) -> List[List[List[int]]]:
    """
    Return a list of sector connections, where each entry describes the connections
    in each layer for that sector. For each layer, a list of two elements describe
    whether there is a connection to the left or right, respectively.
    """
    connections = []
    for i, conn_data_1 in enumerate(contour_connections):
        left = i - 1 if i > 0 else 5
        right = i + 1 if i < len(contour_connections) - 1 else 0

        sector_connections = []
        for ci, (_, conns_in_sector) in enumerate(conn_data_1):
            conns_found = [0] * 2
            for index, conn_data in enumerate([contour_connections[left], contour_connections[right]]):
                for cj, _ in conn_data:
                    if cj in conns_in_sector:
                        conns_found[index] = 1
                        break

            sector_connections.append(conns_found)

        if len(sector_connections) < 6:
            diff = 6 - len(sector_connections)
            sector_connections = sector_connections + ([[0, 0]] * diff)

        connections.append(sector_connections)

    return connections

def group_connections_by_distance(
    connections: List[List[List[int]]],
    distances: List[List[float]]
) -> List[List[List[int]]]:
    """
    Group sector connections by distance to the center, thereby forming layers.
    """
    sector_offsets = [0 for _ in connections]
    for layer in range(5):
        filtered_dists = [(index, dist[min(5, layer + sector_offsets[index])]) for index, dist in enumerate(distances) if dist[min(5, layer + sector_offsets[index])] < MAX_DIST]
        if filtered_dists == []:
            continue

        outliers = []
        prev_dist = filtered_dists[-1][1]
        for index, dist in filtered_dists:
            if dist - prev_dist > DIST_BETWEEN_LAYERS:
                outliers.append(index)

            prev_dist = dist

        for index in outliers:
            new_conns = np.concat((connections[index, :layer, :], np.array([[0, 0]]), connections[index, layer:-1, :]))

            connections[index] = new_conns
            sector_offsets[index] += 1

    return connections

def calc_triangle_area(p1: tuple[float, float], p2: tuple[float, float], p3: tuple[float, float]):
    """
    Calculate the area of the triangle formed by the three given points.
    """
    return abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0)

def get_closest_point_to_mid(contour: MatLike, mid_p: Tuple[int, int]) -> Tuple[int, int]:
    """
    Find the point on the given contour that is closest to the given point `mid_p`.
    """
    min_dist_p = (contour[0][0][0], contour[0][0][1])
    min_dist = MAX_DIST
    for p in contour:
        con_p = (p[0][0], invert_y(p[0][1]))
        dist = eucl_dist(con_p, mid_p)
        if dist < min_dist:
            min_dist = dist
            min_dist_p = con_p

    return min_dist_p

def get_distance_to_obstacle(
    contour: MatLike,
    target_1: Tuple[float, float],
    target_2: Tuple[float, float],
    mid_p: Tuple[int, int]
) -> float:
    """
    Calculate the distance from the center of the middle hexagon
    to the nearest point on the contour of the given obstacle.
    """
    x_target1 = target_1[0]
    y_target1 = target_1[1]
    x_target2 = target_2[0]
    y_target2 = target_2[1]

    bbox = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        cont_x = moments['m10'] / moments['m00']
        cont_y = invert_y(moments['m01'] / moments['m00'])
    else: # Revert to using bounding box
        bbox = cv2.boundingRect(contour)
        cont_x, cont_y = get_mid_bbox(bbox)
        cont_y = invert_y(cont_y)

    hex_y = invert_y(mid_p[1])

    # Calculate the area of four triangles formed by:
    # - target_1, target_2, mid_p (the entire sector)
    # - cont_p, mid_p, target_2
    # - target_1, cont_p, target_2
    # - target_1, mid_p, cont_p
    area = calc_triangle_area((x_target1, y_target1), (mid_p[0], hex_y), (x_target2, y_target2))
    area2 = calc_triangle_area((cont_x, cont_y), (mid_p[0], hex_y), (x_target2, y_target2))
    area3 = calc_triangle_area((x_target1, y_target1), (cont_x, cont_y), (x_target2, y_target2))
    area4 = calc_triangle_area((x_target1, y_target1), (mid_p[0], hex_y), (cont_x, cont_y))

    # Check whether the three smaller triangles (area2, area3, area4) approximate
    # the area of the entire sector, meaning the contour lies within this sector
    # If not, the smaller triangles would make up a bigger area
    dist = MAX_DIST
    cont_area = cv2.contourArea(contour)
    if math.isclose(area, area2 + area3 + area4) and cont_area:
        cont_x, cont_y = get_closest_point_to_mid(contour, mid_p)
        dist = eucl_dist(mid_p, (cont_x, cont_y))

    return dist

def filter_distances(
    distances: List[List[float]],
    connections: List[List[List[int]]],
    areas: List[List[float]]
) -> Tuple[List[List[float]], List[List[List[int]]], List[List[float]]]:
    """
    Filter out distances (and matching connections/areas)
    that are too close to one another. 
    """
    if distances == []:
        return [], [], []

    filtered_dists = [distances[0]]
    filtered_conns = [connections[0]]
    filtered_areas = [areas[0]]
    for dist, conns, area in zip(distances[1:], connections[1:], areas[1:]):
        all_spaced_out = True
        for other_dist in filtered_dists:
            if abs(other_dist - dist) < 15:
                all_spaced_out = False
                break

        if all_spaced_out:
            filtered_dists.append(dist)
            filtered_conns.append(conns)
            filtered_areas.append(area)

    return filtered_dists, filtered_conns, filtered_areas

def sort_features_by_distance(
    distances: List[List[float]],
    connections: List[List[List[int]]],
    areas: List[List[float]]
) -> Tuple[List[List[float]], List[List[List[int]]], List[List[float]]]:
    """
    Sort distances, connections, and areas of obstacles by closest distance to the middle.
    """
    both_sorted = sorted(zip(distances, connections, areas), key=lambda x: x[0])
    return [x[0] for x in both_sorted], [x[1] for x in both_sorted], [x[2] for x in both_sorted]

def get_distances_and_areas_in_sector(
    target1: Tuple[float, float],
    target2: Tuple[float, float],
    mid_p: Tuple[int, int],
    contours: List[MatLike],
    connections: List[List[List[int]]],
    cursor: MatLike
):
    """
    Get all the distances to obstacles, connections between obstacles, and areas
    for the given contours that lie within the sector formed by `target_1`, `target_2`, and `mid_p`
    """
    distances = []
    connected_ids = []
    areas = []
    for index, contour in enumerate(contours):
        if contour is None:
            continue

        dist = get_distance_to_obstacle(contour, target1, target2, mid_p)
        if dist < MAX_DIST:
            distances.append(dist)
            connected_ids.append((index, connections[index]))
            areas.append(cv2.contourArea(contour))

    dist_to_cursor = get_distance_to_obstacle(cursor, target1, target2, mid_p)

    return distances, connected_ids, areas, dist_to_cursor < MAX_DIST

def get_relative_cursor_position(cursor_contour, target_1, target_2, center):
    """
    Get the position of the cursor within the sector it resides.
    This is a value between -1 and 1, where -1 means the cursor is at the
    left most edge of the sector, 0 meaning its in the middle,
    and 1 meaning its at the right most edge.
    """
    dist_cursor_to_center = eucl_dist(cursor_contour, center)
    dist_target_1_to_center = eucl_dist(target_1, center)
    dist_target_2_to_center = eucl_dist(target_2, center)

    dist_1 = eucl_dist(cursor_contour, target_1) - (dist_cursor_to_center - dist_target_1_to_center)
    dist_2 = eucl_dist(cursor_contour, target_2) - (dist_cursor_to_center - dist_target_2_to_center)
    full_dist = dist_1 + dist_2

    left = (dist_2 / full_dist) - 0.5
    right = (dist_1 / full_dist) - 0.5

    if left == right:
        return 0

    max_dist = -left if left > right else right

    return max_dist * 2

def get_obstacle_features(image: MatLike, hexagon: MatLike, cursor_contour: MatLike) -> Optional[Features]:
    """
    For each sector of the hexagon, computes the following things:
    - Distance to all walls/obstacles
    - Connections between walls/obstacles across sectors
    - Areas of walls/obstacles

    Returns these distances in clock-wise order, starting from the sector
    the cursor is currently in. Distances, connections, and areas for each sector
    are sorted by distance, ascending.

    Finally, computes the relative angle of the cursor within its sector.

    All values are saved in and returned as a `Features` object.
    """
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel, iterations=1)

    all_contours = get_contours(image)

    targets = split_screen_into_sectors(image, hexagon)

    filtered_contours = filter_contours(all_contours, hexagon, cursor_contour, False)

    mid_x, mid_y = get_mid_bbox(cv2.boundingRect(hexagon))

    contours = filter_contours(get_contours(image), hexagon, cursor_contour)
    connected_contours = get_contour_connections(filtered_contours, contours)

    distances = [[]] * 6
    obstacles = [[]] * 6
    filtered_connections = []
    cursor_sector = 0
    for i, target in enumerate(targets):
        next_t = targets[0]
        if i < len(targets)-1:
            next_t = targets[i+1]

        dists, conns, areas, has_cursor = get_distances_and_areas_in_sector(
            target, next_t, (mid_x, mid_y), contours, connected_contours, cursor_contour
        )
        filtered_dists, filtered_conns, filtered_areas = filter_distances(dists, conns, areas)
        sorted_dists, sorted_conns, sorted_obstacles = sort_features_by_distance(filtered_dists, filtered_conns, filtered_areas)
        distances[i] = sorted_dists
        filtered_connections.append(sorted_conns)
        obstacles[i] = sorted_obstacles

        if len(distances[i]) < 6:
            diff = 6 - len(distances[i])
            distances[i] = distances[i] + ([MAX_DIST] * diff)

        if len(obstacles[i]) < 6:
            diff = 6 - len(obstacles[i])
            obstacles[i] = obstacles[i] + ([0] * diff)

        if has_cursor:
            cursor_sector = i

    # Get relative position of cursor
    bbox = cv2.boundingRect(cursor_contour)
    mid_cursor = get_mid_bbox(bbox)
    side_1 = hexagon[cursor_sector][0]
    side_2 = hexagon[cursor_sector + 1 if cursor_sector < len(hexagon) - 1 else 0][0]
    cursor_angle = get_relative_cursor_position(mid_cursor, side_1, side_2, (mid_x, mid_y))

    connections = get_sector_connections(filtered_connections)

    distances = np.roll(distances, -cursor_sector, axis=0)
    connections = np.roll(connections, -cursor_sector, axis=0)
    obstacles = np.roll(obstacles, -cursor_sector, axis=0)

    connections = group_connections_by_distance(connections, distances)

    return Features(cursor_angle, distances, connections, obstacles)

def analyze_img(img):
    """
    Helper method for extracting cursor and distances
    from a given image.
    """
    hexagon = extract_hexagon(img)
    thresh_img = get_threshold_img(img, THRESHOLD, True)
    cursor = extract_cursor(thresh_img, get_mid_bbox(cv2.boundingRect(hexagon)))
    if hexagon is None or cursor is None or len(hexagon) != 6:
        return None

    features = get_obstacle_features(thresh_img, hexagon, cursor)

    return features if features and features.distances is not None else None
