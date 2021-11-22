import carla
from third_party.common.vector import Vector3D


class VehicleDimension:
    def __init__(self):
        self._extent = Vector3D()

    @property
    def extent(self):
        return self._extent

    @extent.setter
    def extent(self, var):
        self._extent = var

    @property
    def height(self):
        return self._extent.z * 2

    @property
    def length(self):
        return self._extent.x * 2

    @property
    def width(self):
        return self._extent.y * 2


class VehicleState:
    def __init__(self):
        self._position = Vector3D(0, 0, 0)
        self._heading = 0
        self._length = 0
        self._width = 0

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, var):
        self._position = var

    @property
    def heading(self):
        return self._heading

    @heading.setter
    def heading(self, var):
        self._heading = var

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, var):
        self._length = var

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, var):
        self._width = var

    def clone_carla(self):
        new_instance = VehicleState()
        new_instance.position = carla.Vector3D(
            self._position.x,
            self._position.y,
            self._position.z
        )
        new_instance.heading = self._heading
        new_instance.length = self._length
        new_instance.width = self._width
        return new_instance

    def __repr__(self):
        return f"position: {self._position}\n" \
               f"heading {self._heading}\n" \
               f"length: {self._length}\n" \
               f"width: {self._width}"
