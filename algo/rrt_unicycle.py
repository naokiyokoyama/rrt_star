import magnum as mn
import math
import numpy as np
import quaternion as qt

import habitat_sim

from .utils import PointHeading, heading_to_quaternion, quat_to_rad


def RRTStarUnicycleSelect(rrt_star_parent):
    """Allows for dynamic inheritance to choose which class to inherit from upon instantiation

    :param rrt_star_parent: Must be either RRTStarSim or RRTStarPNG
    :return: RRTStarU class (NOT instance) that inherits from either RRTStarSim or RRTStarPNG.
    """

    class RRTStarU(rrt_star_parent):
        def __init__(
            self,
            params,
            pathfinder,
            critical_angle_lookup=None,
        ):
            """

            :param Any pathfinder: Either a sim.pathfinder or a string (path to an image file)
            :param float max_linear_velocity: Agent's max linear velocity
            :param float max_angular_velocity: Agent's max angular velocity
            :param float near_threshold: How far to look for neighboring points (meters)
            :param float max_distance: How close far-sampled points will be snapped to the closest point
            :param dict critical_angle_lookup: Dict containing critical angles if you already have it
            :param str directory: Where jsons/visualization will be saved
            """
            super().__init__(params, pathfinder)

            # Needed for calculating at what angles the agent should pivot in-place
            # vs. arc toward the next waypoint
            if critical_angle_lookup is None:
                self._critical_angle_lookup = self._generate_critical_angle_lookup()
            else:
                self._critical_angle_lookup = critical_angle_lookup

            self.backwards = params.BACKWARDS_ALLOWED

            # Needed for calculating next waypoints from velocities
            self.vel_control = habitat_sim.physics.VelocityControl()
            self.vel_control.controlling_lin_vel = True
            self.vel_control.controlling_ang_vel = True
            self.vel_control.lin_vel_is_local = True
            self.vel_control.ang_vel_is_local = True

        def _critical_angle(self, euclid_dist, theta=np.pi / 2):
            theta_pivot = 0
            best_path_time = float("inf")
            critical_angle = 0

            while theta >= theta_pivot:
                theta_arc = theta - theta_pivot
                if np.sqrt(1 - np.cos(2 * theta_arc)) < 1e-6:
                    arc_length = euclid_dist
                else:
                    arc_length = (
                        np.sqrt(2)
                        * euclid_dist
                        * theta_arc
                        / np.sqrt(1 - np.cos(2 * theta_arc))
                    )
                arc_time = arc_length / self._max_linear_velocity

                max_arc_turn = arc_time * self._max_angular_velocity

                if max_arc_turn > 2 * theta_arc:
                    pivot_time = theta_pivot / self._max_angular_velocity
                    path_time = arc_time + pivot_time

                    if path_time < best_path_time:
                        best_path_time = path_time
                        critical_angle = theta_arc

                theta_pivot += np.pi / 180.0 * 0.01

            return critical_angle

        def _generate_critical_angle_lookup(self, step_increment=0.01, precision=2):
            critical_angle_lookup = {}
            step = 0.0
            while step <= self._near_threshold:
                step = round(step, precision)
                critical_angle_lookup[step] = self._critical_angle(step)
                step += step_increment
            return critical_angle_lookup

        def _fastest_delta_heading_time(self, theta, euclid_dist, precision=2):
            critical_angle = self._critical_angle_lookup[round(euclid_dist, precision)]

            # How many degrees the robot pivots first
            theta_pivot = max(0, theta - critical_angle)
            pivot_time = theta_pivot / self._max_angular_velocity

            # The difference between the robot's heading and the destination when it starts
            # moving in an arc
            theta_arc = min(critical_angle, theta)

            if np.sqrt(1 - np.cos(2 * theta_arc)) < 1e-6:
                arc_length = euclid_dist
            else:
                arc_length = (
                    np.sqrt(2)
                    * euclid_dist
                    * theta_arc
                    / np.sqrt(1 - np.cos(2 * theta_arc))
                )
            arc_time = arc_length / self._max_linear_velocity

            delta_path_time = arc_time + pivot_time
            delta_heading = theta_pivot + theta_arc * 2

            return delta_heading, delta_path_time

        def _cost_from_to(
            self,
            pt,
            new_pt,
            return_heading=False,
            consider_end_heading=False,
            just_indicate_backwards=False,
        ):
            # theta is the angle from pt to new_pt (0 rad is east)
            theta = math.atan2((new_pt.y - pt.y), new_pt.x - pt.x)
            # theta_diff is angle between the robot's heading at pt to new_pt
            theta_diff = self._get_heading_error(pt.heading, theta)

            # Use backwards motion if faster
            if self.backwards:
                back_heading = self._validate_heading(pt.heading - np.pi)
                theta_diff_back = self._get_heading_error(back_heading, theta)
                if abs(theta_diff_back) < abs(theta_diff):
                    theta_diff = theta_diff_back
                    backwards = True
                else:
                    backwards = False

                if just_indicate_backwards:
                    return backwards

            euclid_dist = self._euclid_2D(pt, new_pt)

            delta_heading, delta_path_time = self._fastest_delta_heading_time(
                abs(theta_diff), euclid_dist
            )

            if return_heading or consider_end_heading:
                if theta_diff < 0:
                    final_heading = pt.heading - delta_heading
                else:
                    final_heading = pt.heading + delta_heading
                final_heading = self._validate_heading(final_heading)
                if consider_end_heading:
                    delta_path_time += abs(
                        self._get_heading_error(new_pt.heading, final_heading)
                        / self._max_angular_velocity
                    )
                if return_heading:
                    return delta_path_time, final_heading

            return delta_path_time

        def _path_exists(self, a, b):
            try:
                intermediate_points = self._get_intermediate_pts(a, b, resolution=0.05)
            except (ValueError, OverflowError):
                return False

            for pt in intermediate_points:
                if not self._is_navigable(pt):
                    return False

            return True

        def _get_intermediate_pts(self, pt, new_pt, precision=2, resolution=0.1):
            """
            Returns points between two waypoints, if there is enough space.
            """
            if (
                self.backwards
                and self._cost_from_to(
                    pt, new_pt, just_indicate_backwards=True
                )
            ):
                pt_heading = self._validate_heading(pt.heading - np.pi)
            else:
                pt_heading = pt.heading

            # theta is the angle from pt to new_pt (0 rad is east)
            theta = math.atan2((new_pt.y - pt.y), new_pt.x - pt.x)
            # theta becomes the angle between the robot's heading at pt to new_pt
            theta_diff = self._get_heading_error(pt_heading, theta)
            theta = abs(theta_diff)
            euclid_dist = self._euclid_2D(pt, new_pt)

            critical_angle = self._critical_angle_lookup[round(euclid_dist, precision)]

            # How many degrees the robot pivots first
            theta_pivot = max(0, theta - critical_angle)

            """
            theta_arc is the difference between the robot's heading and the destination 
            when it starts moving in an arc. Formula for arc_length derived with trigonometry.
            Using trigonometry, we can also prove that the angle between its final heading 
            and the line connecting the start point with the end point is 2*theta_arc.
            """
            theta_arc = min(critical_angle, theta)
            if np.sqrt(1 - np.cos(2 * theta_arc)) < 1e-6:
                arc_length = euclid_dist
            else:
                arc_length = (
                        np.sqrt(2)
                        * euclid_dist
                        * theta_arc
                        / np.sqrt(1 - np.cos(2 * theta_arc))
                )
            arc_time = arc_length / self._max_linear_velocity
            arc_angular_vel = theta_arc * 2 / arc_time

            """
            Determine directions for turning
            """
            if theta_diff < 0:
                theta_pivot = -theta_pivot
                arc_angular_vel = -arc_angular_vel

            """
            We are only interested in the arc_time, because the robot doesn't move
            when pivoting. Get the pivoting out of the way by simple addition.
            """
            pt_pos = pt.as_pos()
            pt_quaternion = heading_to_quaternion(pt_heading + theta_pivot)
            rigid_state = habitat_sim.bindings.RigidState(pt_quaternion, pt_pos)
            self.vel_control.linear_velocity = np.array(
                [0.0, 0.0, -self._max_linear_velocity]
            )  # Always go max linear
            self.vel_control.angular_velocity = np.array([0.0, arc_angular_vel, 0.0])

            num_points = int(round(arc_length / resolution))
            if num_points == 0:
                return []

            all_pts = []
            time_step = arc_time / float(num_points)
            z_step = (new_pt.z - pt.z) / float(num_points)
            for i in range(num_points):
                rigid_state = self.vel_control.integrate_transform(
                    time_step, rigid_state
                )
                rigid_state.translation = mn.Vector3(
                    rigid_state.translation[0],
                    rigid_state.translation[1] + z_step,
                    rigid_state.translation[2],
                )
                end_heading = quat_to_rad(
                    qt.quaternion(
                        rigid_state.rotation.scalar, *rigid_state.rotation.vector
                    )
                )
                end_pt = PointHeading(rigid_state.translation, end_heading)
                all_pts.append(end_pt)

            return all_pts

        def _string_tree(self):
            """
            Return the current graph as a dictionary comprised of strings to save
            to disk.
            """
            string_tree = super()._string_tree()

            fine_path = self.make_path_finer(self._get_best_path())
            fine_path_str = [i._str_key() for i in fine_path]
            string_tree["best_path"] = fine_path_str

            return string_tree

    return RRTStarU
