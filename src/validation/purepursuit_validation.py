import json
import pandas as pd
import numpy as np
from abc import ABC
from typing import Union, Tuple, List

from validation.validation import Validation
from third_party.common.vector import Vector3D
from third_party.common.vehicle_state import VehicleState
from third_party.common.vehicle_dynamics import VehicleDynamics
from third_party.pure_pursuit.pure_pursuit import PurePursuit


class PurePursuitValidation(Validation, ABC):
    def __init__(
            self,
            topology: list,
            map_path: str = None,
    ):
        super().__init__(map_path)

        self.algorithm = PurePursuit(
            topology=topology,
            sampling_distance=0.5,
            min_ld=3.0,
            steer_scale=2.0
        )

    def _separate_input(
            self,
            dynamic_path: str,
            num_inp_timesteps: int = 20
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get input data and ground-truth data
        Those data is separated based on num_inp_timesteps
        Args:
            dynamic_path (str): path to dynamic event
            num_inp_timesteps (int): number of time-steps as input

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]
                - input data frame
                - ground-truth data frame
        """
        data = pd.read_csv(dynamic_path)
        data_by_timestamp = data.groupby(by=[self.ts_col])
        # data container
        inp_df = pd.DataFrame(columns=data.columns)
        gt_df = pd.DataFrame(columns=data.columns)

        for _id, (_ts, _frame) in enumerate(data_by_timestamp):
            # get 20-first-timestamps for input (2s)
            if _id < num_inp_timesteps:
                inp_df = pd.concat([inp_df, _frame], axis=0)
            # the rest is for future prediction (3s)
            else:
                gt_df = pd.concat([gt_df, _frame], axis=0)

        return inp_df, gt_df

    @staticmethod
    def _get_value(
            last_state: pd.DataFrame
    ) -> Tuple[VehicleState, VehicleDynamics]:
        """
        Get last state of object,
        and then convert into VehicleState and VehicleDynamic
        for later usage in module Pure-pursuit
        Args:
            last_state (pd.DataFrame):

        Returns:
            Tuple[VehicleState, VehicleDynamic]
        """
        # get essential data from data frame
        _x, _y = last_state[["center_x", "center_y"]]
        _heading = last_state["heading"]
        _vel = json.loads(last_state["status"])["velocity"]

        # set vehicle state and dynamic
        _vehicle_state = VehicleState()
        _vehicle_state.position = Vector3D(x=_x, y=_y)
        _vehicle_state.heading = _heading
        _vehicle_state.length = 4.0
        _vehicle_dynamic = VehicleDynamics()
        _vehicle_dynamic.velocity.x = _vel

        return _vehicle_state, _vehicle_dynamic

    @staticmethod
    def _fulfill_prediction(
            trajectories: Union[List, np.ndarray],
            num_trajectory: int = 6
    ) -> np.ndarray:
        """
        Purpose of this function is to fulfill dummy trajectories
        to get number of trajectories = num_trajectory(=6)
        (dummy value is very large number: 1e9)
        Args:
            trajectories (Union[List, np.ndarray]): trajectories
            num_trajectory (int): number of trajectory

        Returns:
            (np.ndarray): with shape (N, num_trajectory, T, D)

        """
        if len(trajectories) > num_trajectory:
            return np.array(trajectories)[:num_trajectory]

        num_left = num_trajectory - len(trajectories)
        np_trajs = np.array(trajectories)
        np_trajs_shape = np_trajs.shape
        np_large = np.ones((num_left, np_trajs_shape[1], np_trajs_shape[2])) * 1e9
        return np.concatenate([np_trajs, np_large], axis=0)

    def predict(
            self,
            dynamic_path
    ):
        """
        Get prediction from Pure-pursuit module
        and ground-truth from corresponding sub-scene
        Args:
            dynamic_path (str): path to dynamic event sub-scene

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - pred with shape (N, K, T, 2)
                - gt with shape (N, T, 2)
        """
        inp_df, gt_df = self._separate_input(dynamic_path)

        # get prediction from pure-pursuit
        inp_by_id = inp_df.groupby(by=["id"])
        pred = list()
        for _id, _frame in inp_by_id:
            if _frame["object_type"].iloc[0] == "AV":
                continue
            vehicle_state, vehicle_dynamic = self._get_value(_frame.iloc[-1])
            trajectories = self.algorithm.predict(
                vehicle_state=vehicle_state, vehicle_dynamic=vehicle_dynamic
            )
            np_trajs = self._fulfill_prediction(trajectories, num_trajectory=6)
            pred.append(np_trajs)

        pred = np.array(pred)
        pred = pred[:, :, :, :2]  # (N, K, T, 2)

        # process ground-truth data to numpy with shape (N, T, 2)
        gt = list()
        gt_by_id = gt_df.groupby(by=[self.id_col])
        for _id, _frame in gt_by_id:
            if _frame.iloc[0][self.object_type_col] == "AV":
                continue
            gt.append(_frame[[self.x_col, self.y_col]].to_numpy())
        gt = np.array(gt)
        return pred, gt
