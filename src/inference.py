import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib
from processing.processor import ObjectBehaviorPrediction

matplotlib.use('tkagg')
# logger
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
device = "cuda:0"

# -----
model_path = "models/v4_3/model.30.bin"
map_path = "/home/anhtt163/dataset/OBP/data/inp/static.csv"
data_path = "/home/anhtt163/dataset/OBP/data/inp/inp02.csv"
# -----


if __name__ == "__main__":
    obp = ObjectBehaviorPrediction(model_path, map_path)
    preds = obp.predict(pd.read_csv(data_path))

    # visualize
    _inp = pd.read_csv(data_path, usecols=["id", "timestamp", "center_x", "center_y"])
    _inp_group_by_id = _inp.groupby(by=["id"])
    for _id, frame in _inp_group_by_id:
        _pos = frame[["center_x", "center_y"]].to_numpy()
        plt.scatter(_pos[:, 0], _pos[:, 1], c='b', marker='x', s=10)
        plt.scatter(_pos[-1, 0], _pos[-1, 1], c='g', marker='x', s=30)

    for _id, (trajs, probs) in preds.items():
        # _max_prob_idx = np.argmax(probs)
        # _pred_traj = trajs[_max_prob_idx]
        _pred_traj = trajs.reshape(-1, 2)
        plt.scatter(_pred_traj[:, 0], _pred_traj[:, 1], c='r', marker='o', s=1.)
    plt.show()
