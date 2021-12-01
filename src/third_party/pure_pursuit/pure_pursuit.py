import carla

from third_party.common.graphs import DenseGraph
from third_party.common.vehicle_state import VehicleState
from third_party.common.vehicle_dynamics import VehicleDynamics

from third_party.pure_pursuit.lane_graph import LaneGraph
from third_party.pure_pursuit.path_tracker import PathTracker


class PurePursuit:
    def __init__(
            self,
            topology: list,
            sampling_distance: float = 0.5,
            min_ld: float = 7.0,
            steer_scale: float = 2.0
    ):
        """
        Baseline algorithm for Prediction module
        This class generate trajectories for agent based on
            + Current location
            + Pre-define path/graph
        Args:
            topology (list): list of edges to create graph
            sampling_distance (float): distance between 2 consecutive points in graph
            min_ld (float): distance to look-ahead point
            steer_scale (float): scale value for steering
        """
        self.dense_graph = DenseGraph(
            topology=topology,
            sampling_distance=sampling_distance
        )
        self.lane_graph = LaneGraph(dense_graph=self.dense_graph)
        self.path_tracker = PathTracker(
            min_ld=min_ld,
            steer_scale=steer_scale
        )

    def predict(
            self,
            vehicle_dynamic: VehicleDynamics,
            vehicle_state: VehicleState,
            path_distance: float = 50.,
            trajectory_steps: int = 30,
            trajectory_dt: float = 0.1
    ) -> list:
        """
        Main function to predict trajectories of agent
        Args:
            vehicle_dynamic (VehicleDynamics): vehicle dynamic of agent
            vehicle_state (VehicleState): vehicle state of agent
            path_distance (float): max distance of a pre-defined path
            trajectory_steps (int): number of steps of future trajectory
            trajectory_dt (float): delta time of sampling point

        Returns:
            (list): list(np.ndarray) of possible future trajectories of agent
            Each trajectory will be demonstrated in numpy array with shape (N, 3)
                where   N: number of points (=trajectory_steps)
                        3: 3-dim (x, y, z)
        """
        # create carla.Location
        veh_loc = carla.Location(
            x=vehicle_state.position.x,
            y=vehicle_state.position.y,
            z=vehicle_state.position.z
        )
        # create carla.Rotation
        veh_rot = carla.Rotation(
            yaw=vehicle_state.heading,
            pitch=0,
            roll=0
        )
        # create carla.Transform
        veh_transform = carla.Transform(
            location=veh_loc,
            rotation=veh_rot
        )

        # get all possible paths
        paths = self.lane_graph.get_paths(
            transform=veh_transform,
            distance=path_distance
        )

        # estimate vehicle's trajectory
        trajectories = list()
        for path in paths:
            self.path_tracker.initialize(
                vehicle_dynamic=vehicle_dynamic,
                vehicle_state=vehicle_state
            )
            trajectory = self.path_tracker.track(
                path=path,
                steps=trajectory_steps,
                dt=trajectory_dt
            )
            trajectories.append(trajectory)

        return trajectories
