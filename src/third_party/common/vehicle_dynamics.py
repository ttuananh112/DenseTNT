import carla
import numpy as np
from third_party.common.vector import Vector3D


class VehicleDynamics:
    def __init__(self):
        self._velocity = Vector3D()
        self._angular_velocity = Vector3D()
        self._acceleration = Vector3D()

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, var):
        self._velocity = var

    @property
    def acceleration(self):
        return self._acceleration

    @acceleration.setter
    def acceleration(self, var):
        self._acceleration = var

    @property
    def angular_velocity(self):
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, var):
        self._angular_velocity = var

    def get_speed(self):
        # Somehow .length() doesn't work?
        return np.linalg.norm([self.velocity.x, self.velocity.y])

    def clone_carla(self):
        new_instance = VehicleDynamics()
        new_instance.velocity = carla.Vector3D(
            self._velocity.x,
            self._velocity.y,
            self._velocity.z
        )
        new_instance.acceleration = carla.Vector3D(
            self._acceleration.x,
            self._acceleration.y,
            self._acceleration.z
        )
        new_instance.angular_velocity = carla.Vector3D(
            self._angular_velocity.x,
            self._angular_velocity.y,
            self._angular_velocity.z
        )
        return new_instance

    def __repr__(self):
        return f"velocity: {self._velocity}\n" \
               f"angular_vel: {self._angular_velocity}\n" \
               f"acceleration: {self._acceleration}"
