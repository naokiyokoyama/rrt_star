import numpy as np
import cv2
import math
import os
import habitat_sim

from collections import defaultdict

from .rrt_unicycle import RRTStarUnicycle

class RRTStarPointTurn(RRTStarUnicycle):
    def __init__(
        self,
        pathfinder,
        max_linear_velocity,
        max_angular_velocity,
        near_threshold,
        max_distance,
        goal_minimum_distance=0.2,
        directory=None
    ):
        assert max_distance <= near_threshold, (
            'near_threshold ({}) must be greater than or equal to max_distance ({})'.format(max_distance,
                                                                                            near_threshold)
        )
        self._pathfinder = pathfinder
        self._near_threshold = near_threshold
        self._max_distance = max_distance
        self._max_linear_velocity = max_linear_velocity
        self._max_angular_velocity = max_angular_velocity
        self._goal_minimum_distance = goal_minimum_distance
        self._directory = directory

        if self._directory is not None:
            self._vis_dir = os.path.join(self._directory, 'visualizations')
            self._json_dir = os.path.join(self._directory, 'tree_jsons')
            for i in [self._vis_dir, self._json_dir]:
                if not os.path.isdir(i):
                    os.makedirs(i)

        # TODO: Search the tree with binary search
        self.tree = {}
        self.grid_hash = defaultdict(list)
        self._best_goal_node = None
        self._top_down_img = None
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True
        self._start_iteration = 0
        self._start = None
        self._goal = None
        self._cost_from_parent = {}
        self._shortest_path_points = None
        self.x_min, self.x_max, self.y_min, self.y_max = None, None, None, None

    def _cost_from_to(
        self,
        pt,
        new_pt,
        return_heading=False,
        consider_end_heading=False
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
                    self._get_heading_error(new_pt.heading, final_heading) / self._max_angular_velocity)
            if return_heading:
                return delta_path_time, final_heading

        return delta_path_time

    def _path_exists(self, a, b):
        c = self._pathfinder.try_step_no_sliding(a.as_pos(), b.as_pos())
        return np.allclose(b.as_pos(), c)

    def _string_tree(self):
        """
        Return the current graph as a dictionary comprised of strings to save
        to disk.
        """
        string_graph = {}
        for k, v in self.tree.items():
            if k == self._start:
                string_graph[k._str_key()] = ''
            else:
                string_graph[k._str_key()] = v._str_key()

        string_tree = {
            'start': self._start._str_key(),
            'goal': self._goal._str_key(),
        }

        if self._best_goal_node is not None:
            string_tree['best_goal_node'] = self._best_goal_node._str_key()
            string_tree['best_path_time'] = self._cost_from_start(self._best_goal_node) + self._cost_from_to(
                self._best_goal_node, self._goal)
        else:
            string_tree['best_goal_node'] = ''
            string_tree['best_path_time'] = -1

        # Add the best path
        best_path = self._get_best_path()
        string_tree['best_path_raw'] = [i._str_key() for i in best_path]

        string_tree['graph'] = string_graph

        return string_tree

    def _visualize_tree(
        self,
        meters_per_pixel=0.01,
        show=False,
        path=None,  # Option to visualize another path
        draw_all_edges=True,
        save_path=None
    ):
        '''
        Save and/or visualize the current tree and the best path found so far
        '''
        if self._top_down_img is None:
            self._top_down_img = self.generate_topdown_img(meters_per_pixel=meters_per_pixel)

            # Crop image to just valid points
            mask = cv2.cvtColor(self._top_down_img, cv2.COLOR_BGR2GRAY)
            mask[mask == 255] = 0
            x, y, w, h = cv2.boundingRect(mask)
            self._top_down_img = self._top_down_img[y:y + h, x:x + w]

            # Determine scaling needed
            self._scale_x = lambda x: int((x - self.x_min) / meters_per_pixel)
            self._scale_y = lambda y: int((y - self.y_min) / meters_per_pixel)

        top_down_img = self._top_down_img.copy()

        # Draw all edges in orange
        if draw_all_edges:
            for node, node_parent in self.tree.items():
                if node_parent is None:  # Start point has no parent
                    continue
                cv2.line(
                    top_down_img,
                    (self._scale_x(node.x), self._scale_y(node.y)),
                    (self._scale_x(node_parent.x), self._scale_y(node_parent.y)),
                    (0, 102, 255),
                    1
                )

        # Draw best path to goal if it exists
        if path is not None or self._best_goal_node is not None:
            if path is not None:
                fine_path = path
            else:
                fine_path = self._get_best_path()
            for pt, next_pt in zip(fine_path[:-1], fine_path[1:]):
                cv2.line(
                    top_down_img,
                    (self._scale_x(pt.x), self._scale_y(pt.y)),
                    (self._scale_x(next_pt.x), self._scale_y(next_pt.y)),
                    (0, 255, 0),
                    3
                )

        # Draw start point+heading in blue
        start_x, start_y = self._scale_x(self._start.x), self._scale_y(self._start.y)
        cv2.circle(
            top_down_img,
            (start_x, start_y),
            8,
            (255, 192, 15),
            -1
        )
        LINE_SIZE = 10
        heading_end_pt = (
        int(start_x + LINE_SIZE * np.cos(self._start.heading)), int(start_y + LINE_SIZE * np.sin(self._start.heading)))
        cv2.line(
            top_down_img,
            (start_x, start_y),
            heading_end_pt,
            (0, 0, 0),
            3
        )

        # Draw goal point in red
        # cv2.circle(top_down_img, (self._scale_x(self._goal.x),  self._scale_y(self._goal.y)),  8, (0,255,255), -1)
        SQUARE_SIZE = 6
        cv2.rectangle(
            top_down_img,
            (self._scale_x(self._goal.x) - SQUARE_SIZE, self._scale_y(self._goal.y) - SQUARE_SIZE),
            (self._scale_x(self._goal.x) + SQUARE_SIZE, self._scale_y(self._goal.y) + SQUARE_SIZE),
            (0, 0, 255),
            -1
        )

        # Draw shortest waypoints
        if self._shortest_path_points is None:
            self._get_shortest_path_points()
        for i in self._shortest_path_points:
            cv2.circle(
                top_down_img,
                (self._scale_x(i.x), self._scale_y(i.y)),
                3,
                (255, 192, 15),
                -1
            )

        # Draw fastest waypoints
        if path is None:
            path = self._get_best_path()[1:-1]
        for i in path:
            cv2.circle(
                top_down_img,
                (self._scale_x(i.x), self._scale_y(i.y)),
                3,
                (0, 0, 255),
                -1
            )
            LINE_SIZE = 8
            heading_end_pt = (int(self._scale_x(i.x) + LINE_SIZE * np.cos(i.heading)),
                              int(self._scale_y(i.y) + LINE_SIZE * np.sin(i.heading)))
            cv2.line(
                top_down_img,
                (self._scale_x(i.x), self._scale_y(i.y)),
                heading_end_pt,
                (0, 0, 0),
                1
            )

        if show:
            cv2.imshow('top_down_img', top_down_img)
            cv2.waitKey(1)

        if save_path is not None:
            cv2.imwrite(save_path, top_down_img)

        return top_down_img