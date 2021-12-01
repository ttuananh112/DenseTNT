import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple
from dataset.carla_helper import MapHelper
from utils import rotate, get_subdivide_points


def _get_agent_center_coordinates(
        df_dynamics: pd.DataFrame,
        obj_id: int
) -> Tuple[float, float, float, float]:
    """
    Get current position of AGENT in 2D (x, y, heading, timestamp)
    Args:
        df_dynamics (pd.DataFrame): dataframe of all dynamic objects
        obj_id (int): index of object

    Returns:
        Tuple(float, float, float, float):
            (x, y, heading, timestamp) coordinate of AGENT at moment
    """
    obj_data = df_dynamics.loc[
        df_dynamics["id"] == obj_id,
        ["center_x", "center_y", "heading", "timestamp"]
    ]
    center_x, center_y, heading, timestamp = obj_data.iloc[-1]
    return center_x, center_y, heading, timestamp


def _get_local_map(
        map_helper: MapHelper,
        center_x: float,
        center_y: float,
        heading: float,
) -> Tuple[List, List[np.ndarray]]:
    """
    Get local lane/waypoint in distance (defined in map_helper)
    Args:
        map_helper (MapHelper): support to get local lines
        center_x (float): x-coordinate of AGENT at last recorded time
        center_y (float): y-coordinate of AGENT at last recorded time
        heading (float): orientation of AGENT

    Returns:
        Tuple[List, List[np.ndarray]]:
            - list local map index
            - list of local lines which is non-variant to AGENT
                each line has shape (10, 2) - 10 points, 2 dims x,y
    """
    # # local_map = list(np.ndarray) with np_shape: (10, 2)
    # _x_y = map_helper._map.loc[:, ["x", "y"]].to_numpy()
    # plt.scatter(_x_y[:, 0], _x_y[:, 1], marker='o', c='k', s=10)
    local_id, local_map = map_helper.get_local_lanes(center_x, center_y, heading)

    for line in local_map:
        # plt.scatter(line[:, 0], line[:, 1], marker='o', s=10)
        for i, point in enumerate(line):
            # plt.text(point[0], point[1], i)
            # rotate and translate line for non-variant to AGENT
            point[0], point[1] = rotate(
                point[0] - center_x,
                point[1] - center_y,
                heading
            )
    # plt.scatter(center_x, center_y, marker='v', s=100)
    # plt.show()
    return local_id, local_map


def _get_dynamics_matrix(
        df_dynamics: pd.DataFrame,
        center_x: float,
        center_y: float,
        heading: float,
        timestamp: float
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Function to get dynamics matrix and dynamics spans
    Args:
        df_dynamics (pd.DataFrame): dataframe of dynamics data
        center_x (float): x-coordinate of AGENT
        center_y (float): y-coordinate of AGENT
        heading (float): yaw of AGENT
        timestamp (float): timestamp at moment

    Returns:
        Tuple[np.ndarray, List[Tuple[int, int]]:
            - dynamic_matrix: normalized matrix for input
            - dynamic_spans: list of start and end index of polyline
    """

    def __normalize(row):
        _x, _y = rotate(
            row["center_x"] - center_x,
            row["center_y"] - center_y,
            heading
        )
        row["center_x"] = _x
        row["center_y"] = _y
        row["timestamp"] = row["timestamp"] - timestamp

    # TODO: handle object do not have enough data (2s)
    # normalize by AGENT
    df_dynamics.apply(__normalize, axis=1)
    # group dynamics object by id, type
    object_by_id = df_dynamics.groupby(by=["id"])

    vectors = list()
    dynamics_spans = list()
    for i_object, (_id, frame) in enumerate(object_by_id):
        _start = len(vectors)
        _x_pre = _y_pre = 0.
        for i_state, row in frame.iterrows():
            if i_state > 0:
                vector = np.zeros((128,))
                vector[0] = _x_pre
                vector[1] = _y_pre
                vector[2] = row["center_x"]
                vector[3] = row["center_y"]
                vector[4] = row["timestamp"]
                vector[5] = (row["object_type"] == "AV")
                vector[6] = (row["object_type"] == "AGENT")
                vector[7] = (row["object_type"] == "OTHERS")
                vector[8] = i_object
                vector[9] = i_state
                # add to vector container
                vectors.append(vector)
            # reserve previous position
            _x_pre = row["center_x"]
            _y_pre = row["center_y"]

        _end = len(vectors)
        # mark down start and end index of polyline
        dynamics_spans.append((_start, _end))

    return np.array(vectors), dynamics_spans


def _get_map_matrix(
        map_helper: MapHelper,
        local_index: List,
        list_local_map: List[np.ndarray],
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Function to get map matrix and map spans
    Args:
        map_helper (MapHelper): map helper to get local lines
        local_index (List): list of polyline index
        list_local_map (List[np.ndarray]): list of polyline with shape (N, 2)

    Returns:
        (Tuple[np.ndarray, List[Tuple[int, int]]]):
            - map matrix
            - map spans
    """
    vectors = np.empty((0, 128))
    map_spans = list()
    for i_polygon, (l_idx, points) in enumerate(zip(local_index, list_local_map)):
        _start = len(vectors)
        _x_pre = _y_pre = 0.
        for i_point, point in enumerate(points):
            if i_point > 0:
                vector = np.zeros((128,))
                vector[-1] = _x_pre
                vector[-2] = _y_pre
                vector[-3], vector[-4] = point
                vector[-5] = 1
                vector[-6] = i_point
                vector[-7] = i_polygon
                vector[-8] = -1
                vector[-9] = 0
                vector[-10] = 1 \
                    if map_helper.get_status(l_idx)["intersection"] \
                    else -1

                # estimate pre-pre-point
                if i_point >= 2:
                    point_pre_pre = points[i_point - 2]
                else:
                    point_pre_pre = (
                        2 * _x_pre - point[0],
                        2 * _y_pre - point[1]
                    )
                vector[-17] = point_pre_pre[0]
                vector[-18] = point_pre_pre[1]
                # add to vector container
                vectors = np.concatenate([vectors, vector.reshape(1, -1)])

            # reserve pre-point
            _x_pre, _y_pre = point

        _end = len(vectors)
        # mark down start and end index of polyline
        map_spans.append((_start, _end))

    return vectors, map_spans


def _get_goals_2d(polygons):
    def __get_hash(_point):
        return round((_point[0] + 500) * 100) * 1000000 + \
               round((_point[1] + 500) * 100)

    visit = dict()
    points = list()

    for index_polygon, polygon in enumerate(polygons):
        for i, point in enumerate(polygon):
            hash = __get_hash(point)
            if hash not in visit:
                visit[hash] = True
                points.append(point)

        subdivide_points = get_subdivide_points(polygon)
        points.extend(subdivide_points)

    return np.array(points)


def _get_data(
        mapping: Dict,
        df_dynamics: pd.DataFrame,
        map_helper: MapHelper,
        obj_id: int
):
    """
    Main function to get essential data
        + dynamics data (matrix, spans)
        + map data (matrix, spans)
        + map start index
        + polygon
        + goals 2D
    Args:
        mapping (Dict): data container
        df_dynamics (pd.DataFrame): dataframe of dynamic objects
        map_helper (MapHelper): class to retrieve data from map
        obj_id (int): index of AGENT

    Returns:
        None
    """
    # get center coordinate of AGENT at the moment
    center_x, center_y, heading, timestamp = _get_agent_center_coordinates(
        df_dynamics, obj_id
    )

    # get list local polygon map in distance
    local_idx, list_local_map = _get_local_map(
        map_helper,
        center_x, center_y,
        heading
    )
    # return None if can not find local map
    if len(list_local_map) == 0:
        raise ValueError

    # pre-process dynamics object
    dynamics_matrix, dynamics_spans = _get_dynamics_matrix(
        df_dynamics,
        center_x, center_y,
        heading, timestamp
    )
    # pre-process map
    map_matrix, map_spans = _get_map_matrix(map_helper, local_idx, list_local_map)
    map_spans = [(s + len(dynamics_matrix), e + len(dynamics_matrix)) for s, e in map_spans]
    # concatenate data to matrix
    matrix = np.concatenate([dynamics_matrix, map_matrix], axis=0)
    mapping["matrix"] = matrix

    # polyline_spans
    polyline_spans = dynamics_spans + map_spans
    polyline_spans = [slice(s, e) for s, e in polyline_spans]
    mapping["polyline_spans"] = polyline_spans

    # map_start_polyline_idx
    map_start_polyline_idx = len(dynamics_spans)
    mapping["map_start_polyline_idx"] = map_start_polyline_idx

    # goals_2D
    np_local_map = np.array(list_local_map)
    mapping["polygons"] = np_local_map  # 10 points each polygon
    mapping["goals_2D"] = _get_goals_2d(np_local_map)  # np_local_map.reshape(-1, 2)


def _get_dummy_label(mapping: Dict):
    """
    Function to get dummy label
    This should be redundant in the future
    Args:
        mapping (Dict): data container

    Returns:
        (None)
    """
    mapping["labels"] = np.zeros((30, 2))
    mapping["labels_is_valid"] = np.zeros((30,))
    mapping["stage_one_label"] = 0
    mapping["file_name"] = "dummy_filename"


def preprocess(
        df_dynamics: pd.DataFrame,
        map_helper: MapHelper,
        obj_id: int
):
    """
    Main function to pre-process data
    to feed into model
    Args:
        df_dynamics (pd.DataFrame): dataframe of dynamic objects
        map_helper (MapHelper): class to retrieve data from map
        obj_id (int): index of AGENT

    Returns:
        (Dict): data container
    """
    mapping = dict()
    df_dynamics_clone = df_dynamics.copy(deep=True)

    # assign AGENT for obj_id
    df_dynamics_clone.loc[df_dynamics_clone["object_type"] == "AGENT", "object_type"] = "OTHERS"
    df_dynamics_clone.loc[df_dynamics_clone["id"] == obj_id, "object_type"] = "AGENT"

    # get matrix, polyline_spans and map_start_polyline_idx
    # all value should be normalized by AGENT's position
    try:
        _get_data(mapping, df_dynamics_clone, map_helper, obj_id)
    except ValueError:
        # failed to get data
        return None

    # generate dummy label
    # TODO: this should be redundant...
    _get_dummy_label(mapping)

    return mapping


def postprocess(
        df_dynamics: pd.DataFrame,
        obj_id: int,
        pred_traj: np.ndarray

):
    """
    Main function to post-process data
    to convert predicted data to world coordinates
    Args:
        df_dynamics (pd.DataFrame): dataframe of dynamic objects
        obj_id (int): index of AGENT
        pred_traj (np.ndarray): predicted trajectory (should be (6, 10, 2)

    Returns:
        (np.ndarray): predicted trajectory in world coordinates
    """
    shape = pred_traj.shape
    pred_traj = pred_traj.reshape((-1, 2))
    center_x, center_y, heading, timestamp = _get_agent_center_coordinates(
        df_dynamics, obj_id
    )

    def __de_rotate(x, y, angle):
        _x = x * math.cos(angle) + y * math.sin(angle)
        _y = -x * math.sin(angle) + y * math.cos(angle)
        return _x, _y

    def __denormalize(row):
        _x, _y = __de_rotate(row["center_x"], row["center_y"], heading)
        # translating back to AGENT's world coordinates
        row["center_x"] = _x + center_x
        row["center_y"] = _y + center_y

    df_pred_traj = pd.DataFrame(pred_traj, columns=["center_x", "center_y"])
    df_pred_traj.apply(__denormalize, axis=1)

    return df_pred_traj.to_numpy().reshape(shape)
