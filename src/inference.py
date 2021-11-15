import torch
import argparse
import logging
import utils
import numpy as np

from modeling.vectornet import VectorNet

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
device = "cuda:0"

if __name__ == "__main__":
    model_path = "models/v3/model.50.bin"

    parser = argparse.ArgumentParser()
    utils.add_argument(parser)
    args: utils.Args = parser.parse_args()
    # -----
    # additional args
    args.core_num = 2
    args.do_eval = True
    args.future_frame_num = 30
    args.hidden_size = 128
    args.other_params = [
        "semantic_lane", "direction", "l1_loss", "goals_2D", "enhance_global_graph",
        "subdivide", "lazy_points", "new", "laneGCN", "point_sub_graph", "stage_one",
        "stage_one_dynamic=0.95", "laneGCN-4", "point_level", "point_level-4",
        "point_level-4-3", "complete_traj", "complete_traj-3"
    ]
    args.eval_params = [
        "optimization", "MRminFDE", "cnt_sample=9", "opti_time=0.1"
    ]
    # -----
    utils.init(args, logger)
    model = VectorNet(args).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    mapping = dict()
    # matrix
    matrix = torch.rand((100, 128), device=device)
    mapping["matrix"] = matrix

    # polyline spans
    # dynamic objects
    s = range(0, 41, 20)
    e = range(20, 41, 20)
    polyline_spans = [(i, ii) for i, ii in zip(s, e)]
    # static map
    s = range(40, 101, 10)
    e = range(50, 101, 10)
    polyline_spans += [(i, ii) for i, ii in zip(s, e)]
    # convert to slice
    polyline_spans = [slice(i, ii) for i, ii in polyline_spans]
    mapping["polyline_spans"] = polyline_spans

    # map_start
    # 2 = number of agent polygons
    mapping["map_start_polyline_idx"] = 2

    # labels
    labels = np.zeros([30, 2])
    labels_is_valid = np.ones(args.future_frame_num, dtype=np.int64)
    mapping["labels"] = labels
    mapping["labels_is_valid"] = labels_is_valid

    # goals 2D
    mapping["goals_2D"] = np.random.random((10, 2))
    # 6 polygons, 10 points each polygon, (x, y) coordinates
    polygons = np.random.random((6, 10, 2))
    mapping["polygons"] = polygons.tolist()
    mapping["stage_one_label"] = 0  # index of polygons?

    # file_name
    mapping["file_name"] = "tmp"

    trajs, probs, _ = model([mapping], device)

    print(trajs.shape)
    print(probs.shape)
