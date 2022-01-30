import magnum as mn
import numpy as np
import quaternion as qt

try:
    from habitat_sim.utils.common import quat_rotate_vector
except:
    pass  # allow habitat_sim to not be installed


class PointHeading:
    def __init__(self, point, heading=0.0):
        self.update(point, heading)

    def update(self, point, heading=None):
        self.x, self.z, self.y = point
        self.point = point
        if heading is not None:
            self.heading = heading

    def as_pos(self):
        return np.array([self.x, self.z, self.y]).copy()

    def __key(self):
        return self.x, self.y, self.z, self.heading

    def _str_key(self):
        return "{}_{}_{}_{}".format(self.x, self.y, self.z, self.heading)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, PointHeading):
            return self.__key() == other.__key()
        return NotImplemented


def heading_to_quaternion(heading):
    quat = qt.from_euler_angles([heading + np.pi / 2, 0, 0, 0])
    quat = qt.as_float_array(quat)
    quat = [quat[1], -quat[3], quat[2], quat[0]]
    quat = qt.quaternion(*quat)
    return mn.Quaternion(quat.imag, quat.real)


def quat_to_rad(quat):
    heading_vector = quat_rotate_vector(quat.inverse(), np.array([0, 0, -1]))
    phi = np.arctan2(heading_vector[0], -heading_vector[2])
    return phi - np.pi / 2
