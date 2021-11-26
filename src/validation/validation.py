import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple
from abc import abstractmethod


class Validation:
    def __init__(self, map_path):
        self._map_path = map_path
        self.algorithm = None
        self.mr_epsilon = 2.0  # 2m

        # pre-defined columns from data-frame
        self.ts_col = "timestamp"
        self.id_col = "id"
        self.x_col = "center_x"
        self.y_col = "center_y"
        self.object_type_col = "object_type"

    @abstractmethod
    def predict(
            self,
            dynamic_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Function to get prediction from model
        and ground-truth from dynamics dataframe
        Args:
            dynamic_path (str): path to dynamics dataframe

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - input: shape (N, T', 2)
                - prediction: shape (N, K, T, 2)
                - ground-truth: shape (N, T, 2)
                where,
                    + N: number of objects
                    + K: number of predicted trajectories
                    + T: time-steps in the future
                    + 2: (x, y) coordinates
        """
        pass

    @staticmethod
    def get_fde(
            prediction: np.ndarray,
            ground_truth: np.ndarray
    ) -> np.ndarray:
        """
        Get final-displacement-error
        between prediction and ground-truth
        of each object
        Args:
            prediction (np.ndarray): prediction with shape (N, K, T, 2)
            ground_truth (np.ndarray): ground-truth with shape (N, T, 2)

        Returns:
            (np.ndarray): list of FDE score corresponding to each object
                at current sub-scene, shape = (N,)
        """
        list_fde = list()
        for pred, gt in zip(prediction, ground_truth):
            smallest_displacement = 1e9
            for traj_k in pred:
                # get final position in predictions and ground-truth
                final_pos_pred = traj_k[-1]
                final_pos_gt = gt[-1]
                # calculate the displacement of pred and gt
                displacement = np.linalg.norm(final_pos_gt - final_pos_pred)
                if displacement < smallest_displacement:
                    smallest_displacement = displacement

            # add smallest displacement at final pos in K-predicted-trajectories
            list_fde.append(smallest_displacement)
        return np.array(list_fde)

    def get_mr(
            self,
            fdes: np.ndarray
    ) -> float:
        """
        Check whether prediction is miss or not
        for each object
        Args:
            fdes (np.ndarray): fde array with shape (N,)

        Returns:
            (float): MR score
        """
        mr_score = float(np.sum(fdes > self.mr_epsilon) / len(fdes))
        return mr_score

    def separate_input(
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

    def run(
            self,
            dynamic_folder: str,
            debug: str = None
    ) -> Tuple[float, float]:
        """
        Main function to get validation score
            + mFDE
            + MR
        Args:
            dynamic_folder (str): path to folder
                containing dynamics dataframe
            debug (bool): debugger flag

        Returns:
            Tuple[float, float]:
                + mFDE
                + MR
        """
        fde_container = list()
        for sub_scene_dynamics in glob.glob(f"{dynamic_folder}/*.csv"):
            inp, pred, gt = self.predict(sub_scene_dynamics)

            fde = self.get_fde(pred, gt)
            fde_container.extend(fde)

            if debug is not None:
                # get bad cases
                idx_cases = (np.where(fde > 2)[0]
                             if debug == "bad"
                             else np.where(fde <= 2)[0])
                if len(idx_cases) == 0:
                    continue
                _inp = inp[idx_cases].reshape((-1, 2))
                _pred = pred[idx_cases].reshape((-1, 2))
                _gt = gt[idx_cases].reshape((-1, 2))

                # visualize
                plt.scatter(_inp[:, 0], _inp[:, 1], c="b", s=10, label="inp")
                plt.scatter(_pred[:, 0], _pred[:, 1], c="r", s=10, label="pred")
                plt.scatter(_gt[:, 0], _gt[:, 1], c="g", s=10, label="gt")
                plt.legend(loc="upper right")
                plt.axis("equal")
                plt.show()

        fde_container = np.array(fde_container)
        mean_fde = float(np.mean(fde_container))
        mr_score = self.get_mr(fde_container)
        return mean_fde, mr_score
