import numpy as np


class Vector3D:
    def __init__(
            self,
            carla_vector3d=None,
            x=0, y=0, z=0
    ):
        self._x = x
        self._y = y
        self._z = z
        if carla_vector3d:
            self._x = carla_vector3d.x
            self._y = carla_vector3d.y
            self._z = carla_vector3d.z

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, var):
        self._x = var

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, var):
        self._y = var

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, var):
        self._z = var

    def to_numpy(self):
        return np.array([self._x, self._y, self._z])

    def __add__(self, other):
        v = Vector3D()
        v.x = self._x + other.x
        v.y = self._y + other.y
        v.z = self._z + other.z
        return v

    def __sub__(self, other):
        v = Vector3D()
        v.x = self._x - other.x
        v.y = self._y - other.y
        v.z = self._z - other.z
        return v

    def __mul__(self, other):
        primitive = (int, float)
        v = Vector3D()
        if isinstance(other, Vector3D):
            v.x = self._x * v.x
            v.y = self._y * v.y
            v.z = self._z * v.z
            return v

        elif isinstance(other, primitive):
            v.x = self._x * other
            v.y = self._y * other
            v.z = self._z * other
            return v

        else:
            raise TypeError(f"Type error for {other}, got {type(other)}")

    def __truediv__(self, other):
        primitive = (int, float)
        v = Vector3D()
        if isinstance(other, Vector3D):
            v.x = self._x / v.x
            v.y = self._y / v.y
            v.z = self._z / v.z
            return v

        elif isinstance(other, primitive):
            v.x = self._x / other
            v.y = self._y / other
            v.z = self._z / other
            return v

        else:
            raise TypeError(f"Type error for {other}, got {type(other)}")

    def __eq__(self, other):
        return (self._x == other.x) and (self._y == other.y) and (self._z == other.z)


class Vector2D:
    def __init__(
            self, carla_vector2d=None,
            x=0, y=0
    ):
        self._x = x
        self._y = y

        if carla_vector2d:
            self._x = carla_vector2d.x
            self._y = carla_vector2d.y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, var):
        self._x = var

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, var):
        self._y = var

    def to_numpy(self):
        return np.array([self._x, self._y])

    def __add__(self, other):
        v = Vector2D()
        v.x = self._x + other.x
        v.y = self._y + other.y
        return v

    def __sub__(self, other):
        v = Vector2D()
        v.x = self._x - other.x
        v.y = self._y - other.y
        return v

    def __mul__(self, other):
        primitive = (int, float)
        v = Vector2D()
        if isinstance(other, Vector2D):
            v.x = self._x * v.x
            v.y = self._y * v.y
            return v

        elif isinstance(other, primitive):
            v.x = self._x * other
            v.y = self._y * other
            return v

        else:
            raise TypeError(f"Type error for {other}, got {type(other)}")

    def __truediv__(self, other):
        primitive = (int, float)
        v = Vector2D()
        if isinstance(other, Vector2D):
            v.x = self._x / v.x
            v.y = self._y / v.y
            return v

        elif isinstance(other, primitive):
            v.x = self._x / other
            v.y = self._y / other
            return v

        else:
            raise TypeError(f"Type error for {other}, got {type(other)}")

    def __eq__(self, other):
        return (self._x == other.x) and (self._y == other.y)
