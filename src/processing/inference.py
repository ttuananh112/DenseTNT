import argparse

import utils
import torch
import logging
import pandas as pd

from modeling.vectornet import VectorNet
from concurrent.futures import ProcessPoolExecutor
from dataset.carla_helper import MapHelper
from processing.multiprocess import preprocess, postprocess


class ObjectBehaviorPrediction:
    def __init__(
            self,
            model_path,
            map_path,
            device: str = "cuda:0",
            max_workers: int = 1
    ):
        self._device = device
        self._setup_logging()
        self._setup_args()
        self._load_model(model_path)
        self._map_helper = MapHelper(map_path)

        self._data = []
        self._max_workers = max_workers

    def _setup_logging(self):
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
            datefmt='%m/%d/%Y %H:%M:%S',
            level=logging.INFO
        )
        self._logger = logging.getLogger(__name__)

    def _setup_args(self):
        parser = argparse.ArgumentParser()
        utils.add_argument(parser)
        self._args: utils.Args = parser.parse_args()
        # -----
        # additional args
        self._args.core_num = 2
        self._args.do_eval = True
        self._args.future_frame_num = 30
        self._args.hidden_size = 128
        self._args.other_params = [
            "semantic_lane", "direction", "l1_loss", "goals_2D", "enhance_global_graph",
            "subdivide", "lazy_points", "new", "laneGCN", "point_sub_graph", "stage_one",
            "stage_one_dynamic=0.95", "laneGCN-4", "point_level", "point_level-4",
            "point_level-4-3", "complete_traj", "complete_traj-3"
        ]
        self._args.eval_params = [
            "optimization", "MRminFDE", "cnt_sample=9", "opti_time=0.1"
        ]
        # -----
        utils.init(self._args, self._logger)

    def _load_model(self, model_path):
        self._model = VectorNet(self._args).to(self._device)
        self._model.load_state_dict(torch.load(model_path))
        self._model.eval()

    def predict(
            self,
            df_dynamics: pd.DataFrame,
    ):
        """
        Predict for each object surrounding
        Other objects one by one will be AGENT
        Args:
            df_dynamics (pd.DataFrame):
                DataFrame of all dynamic objects in 20 time-steps (2s)
                columns: ["timestamp", "id", "object_type", "x", "y"]
                                              AV/OTHERS --> AGENT
        """
        list_ids = df_dynamics["id"].unique()
        batch_mapping = list()
        futures = dict()
        result = dict()

        # pre-process data
        # compute multi-process...
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            for _id in list_ids:
                # skip AV data
                if df_dynamics.loc[df_dynamics["id"] == _id, "object_type"].iloc[0] == "AV":
                    continue

                futures[_id] = executor.submit(
                    preprocess,
                    df_dynamics=df_dynamics, map_helper=self._map_helper, obj_id=_id
                )

        ids_to_remove = []
        # get result from processors
        for _id, future in futures.items():
            res = future.result()
            if res is not None:
                batch_mapping.append(res)
            else:
                # reserve to remove later
                ids_to_remove.append(_id)

        # remove object from result
        for _id in ids_to_remove:
            del futures[_id]

        # return if len(batch) == 0
        if len(batch_mapping) == 0:
            return result

        # predict in batch
        trajs, probs, _ = self._model(batch_mapping, self._device)
        # wrap-up into dict, result corresponds to its key
        result = dict(zip(
            futures.keys(),
            [[traj, prob] for traj, prob in zip(trajs, probs)]
        ))

        # post-process
        # compute multi-process
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            for _id in futures.keys():
                # skip AV data
                if df_dynamics.loc[df_dynamics["id"] == _id, "object_type"].iloc[0] == "AV":
                    continue

                futures[_id] = executor.submit(
                    postprocess,
                    df_dynamics=df_dynamics, obj_id=_id, pred_traj=result[_id][0]
                )

        # get result
        for _id, future in futures.items():
            result[_id][0] = future.result()

        return result
