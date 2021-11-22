import pandas as pd
import numpy as np

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

    def _separate_input(self, dynamic_path, num_timesteps=20):
        data = pd.read_csv(dynamic_path)
        data_by_timestamp = data.groupby(by=[self.ts_col])
        # data container
        inp_df = pd.DataFrame(columns=data.columns)
        gt_df = pd.DataFrame(columns=data.columns)

        for _id, (_ts, _frame) in enumerate(data_by_timestamp):
            # get 20-first-timestamps for input (2s)
            if _id < num_timesteps:
                inp_df = pd.concat([inp_df, _frame], axis=0)
            # the rest is for future prediction (3s)
            else:
                gt_df = pd.concat([gt_df, _frame], axis=0)

        return inp_df, gt_df

    def predict(self, dynamic_path):
        inp_df, gt_df = self._separate_input(dynamic_path)

        # get prediction from model
        out = self.algorithm.predict(inp_df)
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
        return pred, gt
