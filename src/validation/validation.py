import glob
import numpy as np
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function to get prediction from model
        and ground-truth from dynamics dataframe
        Args:
            dynamic_path (str): path to dynamics dataframe

        Returns:
            Tuple[np.ndarray, np.ndarray]:
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
        mr_score = float(np.sum(fdes <= self.mr_epsilon) / len(fdes))
        return mr_score

    def run(
            self,
            dynamic_folder: str
    ) -> Tuple[float, float]:
        """
        Main function to get validation score
            + mFDE
            + MR
        Args:
            dynamic_folder (str): path to folder
                containing dynamics dataframe

        Returns:
            Tuple[float, float]:
                + mFDE
                + MR
        """
        fde_container = list()
        for sub_scene_dynamics in glob.glob(f"{dynamic_folder}/*.csv"):
            pred, gt = self.predict(sub_scene_dynamics)
            fde_container.extend(self.get_fde(pred, gt))

        fde_container = np.array(fde_container)
        mean_fde = float(np.mean(fde_container))
        mr_score = self.get_mr(fde_container)
        return mean_fde, mr_score
