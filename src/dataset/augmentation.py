import numpy as np
from typing import Dict


def flip_vertically(mapping: Dict):
    matrix = mapping["matrix"]
    polyline_idx = mapping["map_start_polyline_idx"]
    polyline_spans = mapping["polyline_spans"]

    # MATRIX
    map_start_idx = polyline_spans[polyline_idx].start
    actor_vectors = matrix[:map_start_idx]
    map_vectors = matrix[map_start_idx:]

    # flip actors' y-coordinate trajectory
    actor_vectors[:, 1] = -actor_vectors[:, 1]
    actor_vectors[:, 3] = -actor_vectors[:, 3]

    # flip map polylines y-coordinate position
    map_vectors[:, -2] = -map_vectors[:, -2]
    map_vectors[:, -4] = -map_vectors[:, -4]
    map_vectors[:, -18] = -map_vectors[:, -18]

    mapping["matrix"] = np.concatenate([actor_vectors, map_vectors], axis=0)

    # POLYGON
    polygons = np.array(mapping["polygons"])
    polygons[:, 1] = -polygons[:, 1]
    mapping["polygons"] = polygons.tolist()

    # GOALS_2D
    mapping["goals_2D"][:, 1] = -mapping["goals_2D"][:, 1]

    # LABELS
    mapping["labels"][:, 1] = -mapping["labels"][:, 1]

    return mapping
