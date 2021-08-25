import cv2
import numpy as np
import math


def RRTStarPTSelect(rrt_star_parent):
    """Allows for dynamic inheritance to choose which class to inherit from upon instantiation

    :param rrt_star_parent: Must be either RRTStarSim or RRTStarPNG
    :return: RRTStarPT class (NOT instance) that inherits from either RRTStarSim or RRTStarPNG.
    """

    class RRTStarPT(rrt_star_parent):
        def _cost_from_to(
            self, pt, new_pt, return_heading=False, consider_end_heading=False
        ):
            # theta is the angle from pt to new_pt (0 rad is east)
            theta = math.atan2((new_pt.y - pt.y), new_pt.x - pt.x)
            # theta_diff is angle between the robot's heading at pt to new_pt
            theta_diff = self._get_heading_error(pt.heading, theta)
            euclid_dist = self._euclid_2D(pt, new_pt)

            pivot_time = abs(theta_diff) / self._max_angular_velocity
            move_time = abs(euclid_dist) / self._max_linear_velocity

            delta_path_time = pivot_time + move_time

            if return_heading or consider_end_heading:

                final_heading = pt.heading + theta_diff

                if final_heading > np.pi:
                    final_heading -= np.pi * 2
                elif final_heading < -np.pi:
                    final_heading += np.pi * 2

                if consider_end_heading:
                    delta_path_time += abs(
                        self._get_heading_error(new_pt.heading, final_heading)
                        / self._max_angular_velocity
                    )
                if return_heading:
                    return delta_path_time, final_heading

            return delta_path_time

        def _path_exists(self, a, b):
            if self.pathfinder_type == 'habitat_sim':
                c = self._pathfinder.try_step_no_sliding(a.as_pos(), b.as_pos())

                return np.allclose(b.as_pos(), c)

            elif self.pathfinder_type == 'png':
                '''
                Draw a straight white line on the black-white map.
                If the resulting map is different, a wall has been crossed.
                '''
                map_copy = self._map.copy()

                cv2.line(
                    map_copy,
                    (self._scale_x(a.x), self._scale_y(a.y)),
                    (self._scale_x(b.x), self._scale_y(b.y)),
                    255, # color (white)
                    1, # thickness
                )

                return not np.bitwise_xor(map_copy, self._map.copy()).any()

            else:
                raise NotImplementedError(f'Pathfinder type {self.pathfinder_type} not supported.')

        def _get_intermediate_pts(self, *args, **kwargs):
            # Intermediate points are not necessary for point-turn behavior
            return []

    return RRTStarPT
