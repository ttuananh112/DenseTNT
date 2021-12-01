import numpy as np
from typing import Tuple

from abc import ABC
from validation.validation import Validation
from inference import ObjectBehaviorPrediction


class DenseTNTValidation(Validation, ABC):
    def __init__(
            self,
            map_path: str,
            model_path: str,
            max_workers: int = 5
    ):
        super().__init__(map_path)
        self.algorithm = ObjectBehaviorPrediction(
            model_path, self._map_path, max_workers=max_workers
        )

    def predict(
            self,
            dynamic_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get prediction from DenseTNT model
        and ground-truth from corresponding sub-scene
        Args:
            dynamic_path (str): path to dynamic event sub-scene

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - inp with shape (N, T', 2)
                - pred with shape (N, K, T, 2)
                - gt with shape (N, T, 2)
        """
        inp_df, gt_df = self.separate_input(dynamic_path)

        # process inp to numpy with shape (N, T', 2)
        inp = list()
        for _id, _frame in inp_df.groupby(by=[self.id_col]):
            if _frame.iloc[0][self.object_type_col] == "AV":
                continue
            inp.append(_frame[[self.x_col, self.y_col]].to_numpy())
        inp = np.array(inp)

        # get prediction from model
        out = self.algorithm.predict(inp_df)
        # return zeros if can not produce input
        if not out:
            return np.zeros((1, 20, 2)), np.zeros((1, 1, 30, 2)), np.zeros((1, 30, 2))

        pred = [traj for traj, _ in out.values()]
        pred = np.array(pred)  # pred should have shape (N, K, T, 2)

        # process ground-truth data to numpy with shape (N, T, 2)
        gt = list()
        gt_by_id = gt_df.groupby(by=[self.id_col])
        for _id, _frame in gt_by_id:
            if _frame.iloc[0][self.object_type_col] == "AV":
                continue
            gt.append(_frame[[self.x_col, self.y_col]].to_numpy())
        gt = np.array(gt)
        return inp, pred, gt
