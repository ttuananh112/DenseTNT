import math
import numpy as np
from third_party.common.vehicle_dynamics import VehicleDynamics
from third_party.common.vehicle_state import VehicleState
from third_party.common.utils import norm, vector3d_to_numpy


class PathTracker:

    def __init__(
            self,
            min_ld: float = 7.0,
            steer_scale: float = 2.0
    ):
        """
        Object to generate trajectory for ego vehicle with pre-defined path given
        Args:
            min_ld (float): min look-ahead distance
            steer_scale (float): value to scale steering angle
        """
        self.min_ld = min_ld
        self.steer_scale = steer_scale

        self.path = None
        self.point_tracker = None

    def initialize(
            self,
            vehicle_dynamic: VehicleDynamics,
            vehicle_state: VehicleState
    ):
        """
        Initialize PointTracker
        Args:
            vehicle_dynamic (VehicleDynamic): VehicleDynamic
            vehicle_state (VehicleState): VehicleState

        Returns:

        """
        veh_dynamic = vehicle_dynamic.clone_carla()
        veh_state = vehicle_state.clone_carla()
        self.point_tracker = PointTracker(veh_dynamic, veh_state, self.steer_scale)

    def track(
            self,
            path: np.ndarray,
            steps: int,
            dt: float = 0.5
    ) -> np.ndarray:
        """
        Get trajectory in S steps forward
        Args:
            path (np.ndarray): pre-defined path in numpy array with shape (N, 2)
            steps (int): number of points in final trajectory
            dt: (float): delta time between 2 steps

        Returns
            (np.ndarray): trajectory with shape (steps, 2)
        """
        self.path = path.copy()
        trajectory = []
        for _ in range(steps):
            target_point = self.get_target_point()

            point = self.point_tracker.track(target_point, dt).tolist()
            trajectory.append(point)

        return np.array(trajectory)

    def get_target_point(self) -> np.ndarray:
        """
        Find target point base on min_ld,
        the distance from ego car to target point should >= min_ld

        Return:
            (np.ndarray): target point in numpy [x, y, z]
        """
        curr_pos_numpy = vector3d_to_numpy(self.point_tracker.state.position)
        # get the target_point which has distance >= self.min_ld
        while norm(self.path[0] - curr_pos_numpy) < self.min_ld:

            # if there is only one point left in self.path
            # or agent could go through all points in self.path
            # -> create new point in front of agent
            if len(self.path) == 1:
                pos_heading = np.array([
                    np.cos(self.point_tracker.state.heading),
                    np.sin(self.point_tracker.state.heading),
                    0
                ])
                return curr_pos_numpy + pos_heading

            # pop the first item
            self.path = self.path[1:]
        return self.path[0]


class PointTracker:

    def __init__(
            self,
            vehicle_dynamic: VehicleDynamics,
            vehicle_state: VehicleState,
            steer_scale: float
    ):
        """
        Object to generate the position of ego vehicle
        in order to reach the target point, based on pure pursuit algorithm
        Args:
            vehicle_dynamic (VehicleDynamics): VehicleDynamics
            vehicle_state (VehicleDynamics): VehicleDynamics
            steer_scale (float): scale value for steering angle
        """
        self.dynamics = vehicle_dynamic
        # using scalar of velocity and acceleration instead of vector
        self.dynamics.velocity = norm(vector3d_to_numpy(self.dynamics.velocity))
        self.dynamics.acceleration = norm(vector3d_to_numpy(self.dynamics.acceleration))

        self.state = vehicle_state
        self.steer_scale = steer_scale
        self.steer = 0

    def track(
            self,
            target_point,
            dt
    ) -> np.ndarray:
        """
        Get position of ego vehicle to reach target point after dt time
        Args:
            target_point (np.ndarray): target point in numpy array
            dt (float): delta time

        Returns:
            (np.ndarray): position of ego vehicle after delta time
        """
        self.get_steering_angle(target_point)
        self.update(dt=dt)

        return vector3d_to_numpy(self.state.position)

    def get_steering_angle(
            self,
            target_point: np.ndarray
    ) -> None:
        """
        Apply pure pursuit algorithm to find steering angle,
        that leads ego vehicle to target point.
        Result will be stored in self.steer
        Args:
            target_point (np.ndarray): target point

        Returns:
            (None)
        """
        target_vector = target_point - vector3d_to_numpy(self.state.position)
        ld = norm(target_vector)

        alpha = math.atan2(target_vector[1], target_vector[0]) - self.state.heading
        kappa = (2 * math.sin(alpha)) / ld

        # steering angle should be filter throw a better function, other than simple scale-up function
        self.steer = math.atan2(kappa * self.state.length, 1) * self.steer_scale
        # need to clip between -1 and 1
        self.steer = np.clip(self.steer, -1, 1)

    def update(
            self,
            dt: float = 0.5
    ) -> None:
        """
        Update VehicleDynamic and VehicleState of ego vehicle after dt time
        Args:
            dt (float): delta time

        Returns:
            (None)
        """
        self.dynamics.velocity += self.dynamics.acceleration * dt
        self.state.position.x += (self.dynamics.velocity * math.cos(self.state.heading) * dt) + (
                0.5 * self.dynamics.acceleration * math.cos(self.state.heading) * dt ** 2)
        self.state.position.y += (self.dynamics.velocity * math.sin(self.state.heading) * dt) + (
                0.5 * self.dynamics.acceleration * math.sin(self.state.heading) * dt ** 2)
        self.state.heading += self.dynamics.velocity * (math.tan(self.steer) / self.state.length) * dt
