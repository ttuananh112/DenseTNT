import json

import carla
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from third_party.common.vector import Vector3D
from third_party.common.vehicle_state import VehicleState
from third_party.common.vehicle_dynamics import VehicleDynamics

from third_party.pure_pursuit.pure_pursuit import PurePursuit

HOST = "127.0.0.1"
PORT = 2000
TOWN = "Town01"
DATAFRAME = "/home/anhtt163/dataset/OBP/datav4/inp/inp02.csv"

CLIENT = carla.Client(host=HOST, port=PORT)
WORLD = CLIENT.load_world(TOWN)
MAP = WORLD.get_map()
TOPOLOGY = MAP.get_topology()


def get_value(last_state):
    _x, _y = last_state[["center_x", "center_y"]]
    _heading = last_state["heading"]
    _vel = json.loads(last_state["status"])["velocity"]

    _vehicle_state = VehicleState()
    _vehicle_state.position = Vector3D(x=_x, y=_y)
    _vehicle_state.heading = _heading
    _vehicle_state.length = 4.0
    _vehicle_dynamic = VehicleDynamics()
    _vehicle_dynamic.velocity.x = _vel

    return _vehicle_state, _vehicle_dynamic


if __name__ == "__main__":
    pure_pursuit = PurePursuit(
        topology=TOPOLOGY,
        sampling_distance=0.5,
        min_ld=3.0,
        steer_scale=2.0
    )

    data = pd.read_csv(DATAFRAME)
    data_by_id = data.groupby(by=["id"])
    for _id, _frame in data_by_id:
        if _frame["object_type"].iloc[0] == "AV":
            continue

        vehicle_state, vehicle_dynamic = get_value(_frame.iloc[-1])

        trajectories = pure_pursuit.predict(
            vehicle_state=vehicle_state, vehicle_dynamic=vehicle_dynamic
        )
        np_trajs = np.array(trajectories)

        for traj in np_trajs:
            plt.plot(traj[:, 0], traj[:, 1])

    plt.axis("equal")
    plt.show()
