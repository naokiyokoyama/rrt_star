import cv2
import numpy as np
import tqdm
import math
import time
import json
import os
import glob
import magnum as mn
import random
from collections import defaultdict

import habitat_sim
from habitat_sim.nav import ShortestPath

from .utils import PointHeading, heading_to_quaternion, quat_to_rad

class RRTStarUnicycle:
    def __init__(
        self,
        pathfinder,
        max_linear_velocity,
        max_angular_velocity,
        near_threshold,
        max_distance,
        goal_minimum_distance=0.2,
        critical_angle_lookup=None,
        directory=None
    ):
        assert near_threshold >= max_distance, (
            f'near_threshold ({near_threshold}) must be greater than or'
            f'equal to max_distance ({max_distance})'
        )
        self._pathfinder = pathfinder
        self._near_threshold = near_threshold
        self._max_distance = max_distance
        self._max_linear_velocity = max_linear_velocity
        self._max_angular_velocity = max_angular_velocity
        self._goal_minimum_distance = goal_minimum_distance
        self._directory = directory

        if critical_angle_lookup is None:
            self._critical_angle_lookup = self._generate_critical_angle_lookup()
        else:
            self._critical_angle_lookup = critical_angle_lookup

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

    def _euclid_2D(self, p1, p2):
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def _is_navigable(self, pt, max_y_delta=0.5):
        return self._pathfinder.is_navigable(pt.as_pos(), max_y_delta=max_y_delta)

    def _critical_angle(self, euclid_dist, theta=np.pi/2):
        theta_pivot = 0
        best_path_time = float('inf')
        critical_angle = 0

        while theta >= theta_pivot:
            theta_arc = theta - theta_pivot
            if np.sqrt(1 - np.cos(2 * theta_arc)) < 1e-6:
                arc_length = euclid_dist
            else:
                arc_length = np.sqrt(2) * euclid_dist * theta_arc / np.sqrt(1 - np.cos(2 * theta_arc))
            arc_time = arc_length / self._max_linear_velocity

            max_arc_turn = arc_time * self._max_angular_velocity

            if max_arc_turn > 2 * theta_arc:
                pivot_time = theta_pivot / self._max_angular_velocity
                path_time = arc_time + pivot_time

                if path_time < best_path_time:
                    best_path_time = path_time
                    critical_angle = theta_arc

            theta_pivot += np.pi / 180. * 0.01

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
            arc_length = np.sqrt(2) * euclid_dist * theta_arc / np.sqrt(1 - np.cos(2 * theta_arc))
        arc_time = arc_length / self._max_linear_velocity

        delta_path_time = arc_time + pivot_time
        delta_heading = theta_pivot + theta_arc * 2

        return delta_heading, delta_path_time

    def _get_heading_error(self, source, target):
        diff = target - source
        if diff > np.pi:
            diff -= np.pi * 2
        elif diff < -np.pi:
            diff += np.pi * 2
        return diff

    def _get_path_to_start(self, pt):
        path = [pt]
        while pt != self._start:
            path.insert(0, self.tree[pt])
            pt = self.tree[pt]
        return path

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

        delta_heading, delta_path_time = self._fastest_delta_heading_time(abs(theta_diff), euclid_dist)

        if return_heading or consider_end_heading:
            if theta_diff < 0:
                final_heading = pt.heading - delta_heading
            else:
                final_heading = pt.heading + delta_heading
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

    def _cost_from_start(self, pt):
        path = self._get_path_to_start(pt)
        cost = 0
        for parent, child in zip(path[:-1], path[1:]):
            if child not in self._cost_from_parent:
                self._cost_from_parent[child] = self._cost_from_to(
                    parent,
                    child,
                    consider_end_heading=True
                )
            cost += self._cost_from_parent[child]
        return cost

    def _max_point(self, p1, p2):
        euclid_dist = self._euclid_2D(p1, p2)
        if euclid_dist <= self._max_distance:
            return p2, False

        new_x = p1.x + (p2.x - p1.x * self._max_distance / euclid_dist)
        new_y = p1.y + (p2.y - p1.y * self._max_distance / euclid_dist)
        new_z = p1.z
        p_new = PointHeading(self._pathfinder.snap_point([new_x, new_z, new_y]))  # MAY RETURN NAN NAN NAN

        return p_new, True

    def _get_near_pts(self, pt):
        ret = []
        i = int((pt.x - self.x_min) // self._near_threshold)
        j = int((pt.y - self.y_min) // self._near_threshold)
        ret += self.grid_hash[(i, j)]
        left = ((pt.x - self.x_min) % self._near_threshold) < self._near_threshold / 2
        down = ((pt.y - self.y_min) % self._near_threshold) < self._near_threshold / 2
        if left:
            ret += self.grid_hash[(i - 1, j)]
            if down:
                ret += self.grid_hash[(i - 1, j - 1)]
                ret += self.grid_hash[(i, j - 1)]
            else:
                ret += self.grid_hash[(i - 1, j + 1)]
                ret += self.grid_hash[(i, j + 1)]
        else:
            ret += self.grid_hash[(i + 1, j)]
            if down:
                ret += self.grid_hash[(i + 1, j - 1)]
                ret += self.grid_hash[(i, j - 1)]
            else:
                ret += self.grid_hash[(i + 1, j + 1)]
                ret += self.grid_hash[(i, j + 1)]

        return ret

    def _closest_tree_pt(self, pt):
        neighbors = []
        i = int((pt.x - self.x_min) // self._near_threshold)
        j = int((pt.y - self.y_min) // self._near_threshold)
        count = 0
        nearby_grids = [(i, j)]
        while not neighbors:
            if count > 0:
                for c in range(-count + 1, count):
                    nearby_grids.append((i + count, j + c))  # Right grids
                    nearby_grids.append((i - count, j + c))  # Left grids
                    nearby_grids.append((i + c, j + count))  # Upper grids
                    nearby_grids.append((i + c, j - count))  # Lower grids
                # Corner grids
                nearby_grids.append((i + count, j + count))
                nearby_grids.append((i + count, j - count))
                nearby_grids.append((i - count, j + count))
                nearby_grids.append((i - count, j - count))
            for ii, jj in nearby_grids:
                neighbors += self.grid_hash[(ii, jj)]
            count += 1
            nearby_grids = []

        return min(neighbors, key=lambda x: self._euclid_2D(pt, x))

    def _path_exists(self, a, b):
        try:
            intermediate_points = self._get_intermediate_pts(a, b, resolution=0.05)
        except (ValueError, OverflowError):
            return False

        for pt in intermediate_points:
            if not self._is_navigable(pt):
                return False

        return True

    def make_path_finer(self, path, precision=2, resolution=0.01):
        all_pts = []
        for pt, new_pt in zip(path[:-1], path[1:]):
            all_pts += self._get_intermediate_pts(pt, new_pt, precision=precision, resolution=resolution)

        return all_pts

    def _get_intermediate_pts(self, pt, new_pt, precision=2, resolution=0.1):
        '''
        Only return the points between, if there is enough space.
        '''

        # theta is the angle from pt to new_pt (0 rad is east)
        theta = math.atan2((new_pt.y - pt.y), new_pt.x - pt.x)
        # theta becomes the angle between the robot's heading at pt to new_pt
        theta_diff = self._get_heading_error(pt.heading, theta)
        theta = abs(theta_diff)
        euclid_dist = self._euclid_2D(pt, new_pt)

        critical_angle = self._critical_angle_lookup[round(euclid_dist, precision)]

        # How many degrees the robot pivots first
        theta_pivot = max(0, theta - critical_angle)
        pivot_time = theta_pivot / self._max_angular_velocity

        '''
        theta_arc is the difference between the robot's heading and the destination 
        when it starts moving in an arc. Formula for arc_length dervied with trigonometry.
        Using trigonometry, we can also prove that the angle between its final heading 
        and the line connecting the start point with the end point is 2*theta_arc.
        '''
        theta_arc = min(critical_angle, theta)
        if np.sqrt(1 - np.cos(2 * theta_arc)) < 1e-6:
            arc_length = euclid_dist
        else:
            arc_length = np.sqrt(2) * euclid_dist * theta_arc / np.sqrt(1 - np.cos(2 * theta_arc))
        arc_time = arc_length / self._max_linear_velocity
        arc_angular_vel = theta_arc * 2 / arc_time

        '''
        Determine directions for turning
        '''
        if theta_diff < 0:
            theta_pivot = -theta_pivot
            arc_angular_vel = -arc_angular_vel

        '''
        We are only interested in the arc_time, because the robot doesn't move
        when pivoting. Get the pivoting out of the way by simple addition.
        '''
        pt_pos = pt.as_pos()
        pt_quaternion = heading_to_quaternion(pt.heading + theta_pivot)
        rigid_state = habitat_sim.bindings.RigidState(pt_quaternion, pt_pos)
        self.vel_control.linear_velocity = np.array([0.0, 0.0, -self._max_linear_velocity])  # Always go max linear
        self.vel_control.angular_velocity = np.array([0., arc_angular_vel, 0.])

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
                rigid_state.translation[2]
            )
            end_heading = quat_to_rad(
                np.quaternion(rigid_state.rotation.scalar,
                              *rigid_state.rotation.vector)
            ) - np.pi / 2
            end_pt = PointHeading(rigid_state.translation, end_heading)
            all_pts.append(end_pt)

        return all_pts

    def _str_to_pt(self, pt_str):
        x, y, z, heading = pt_str.split('_')
        point = (float(x), float(z), float(y))
        return PointHeading(point, float(heading))

    def _load_tree_from_json(self, json_path=None):
        '''
        Attempts to recover the latest existing tree from the json directory,
        or provided json_path.
        '''
        if json_path is None:
            existing_jsons = glob.glob(os.path.join(self._json_dir, '*.json'))
            if len(existing_jsons) > 0:
                get_num_iterations = lambda x: int(os.path.basename(x).split('_')[0])
                json_path = sorted(existing_jsons, key=get_num_iterations)[-1]
            else:
                return None

        with open(json_path) as f:
            latest_string_tree = json.load(f)

        start_str = latest_string_tree['start']
        if self._start is None:
            self._start = self._str_to_pt(start_str)

        goal_str = latest_string_tree['goal']
        if self._goal is None:
            self._goal = self._str_to_pt(goal_str)

        if self.x_min is None:
            self._set_offsets()

        for k, v in latest_string_tree['graph'].items():
            pt = self._str_to_pt(k)
            if k == latest_string_tree['best_goal_node']:
                self._best_goal_node = pt

            if v == '':  # start node is key
                continue
            if v == start_str:
                self.tree[pt] = self._start
                self.add_to_grid_hash(pt)
            else:
                pt_v = self._str_to_pt(v)
                self.tree[pt] = pt_v
                self.add_to_grid_hash(pt)

        self._start_iteration = int(os.path.basename(json_path).split('_')[0]) + 1

        return None

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
                fine_path = [node_parent] + self._get_intermediate_pts(node_parent, node, resolution=0.01) + [node]
                for pt, next_pt in zip(fine_path[:-1], fine_path[1:]):
                    cv2.line(
                        top_down_img,
                        (self._scale_x(pt.x), self._scale_y(pt.y)),
                        (self._scale_x(next_pt.x), self._scale_y(next_pt.y)),
                        (0, 102, 255),
                        1
                    )

        # Draw best path to goal if it exists
        if path is not None or self._best_goal_node is not None:
            if path is not None:
                fine_path = self.make_path_finer(path)
            else:
                fine_path = self.make_path_finer(self._get_best_path())
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

    def _get_best_path(self):
        if self._best_goal_node is None:
            return []
        return self._get_path_to_start(self._best_goal_node) + [self._goal]

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
        fine_path = self.make_path_finer(best_path)
        fine_path_str = [i._str_key() for i in fine_path]
        string_tree['best_path'] = fine_path_str
        string_tree['best_path_raw'] = [i._str_key() for i in best_path]

        string_tree['graph'] = string_graph

        return string_tree

    def add_to_grid_hash(self, pt):
        i = int((pt.x - self.x_min) // self._near_threshold)
        j = int((pt.y - self.y_min) // self._near_threshold)
        self.grid_hash[(i, j)].append(pt)

    def _set_offsets(self):
        self.x_min, self.y_min = float('inf'), float('inf')
        self.x_max, self.y_max = float('-inf'), float('-inf')
        for v in self._pathfinder.build_navmesh_vertices():
            pt = PointHeading(v)
            # Make sure it's on the same elevation as the start point
            if abs(pt.z - self._start.z) < 0.8:
                self.x_min = min(self.x_min, pt.x)
                self.y_min = min(self.y_min, pt.y)
                self.x_max = max(self.x_max, pt.x)
                self.y_max = max(self.y_max, pt.y)

    def _get_shortest_path_points(self):
        sp = ShortestPath()
        sp.requested_start = self._start.as_pos()
        sp.requested_end = self._goal.as_pos()
        self._pathfinder.find_path(sp)
        self._shortest_path_points = [PointHeading(i) for i in sp.points]

    def generate_tree(
            self,
            start_position,
            start_heading,
            goal_position,
            json_path=None,
            iterations=5e4,
            visualize_on_screen=False,
            visualize_iterations=500,
            seed=0
    ):
        np.random.seed(seed)
        random.seed(seed)
        self._start = PointHeading(start_position, heading=start_heading)
        self._goal = PointHeading(goal_position)
        self.tree[self._start] = None
        self._cost_from_parent[self._start] = 0
        # self.times = defaultdict(list)

        self._get_shortest_path_points()

        self._set_offsets()

        self.add_to_grid_hash(self._start)
        self._load_tree_from_json(json_path=json_path)

        for iteration in tqdm.trange(int(iterations + 1)):
            # for iteration in range(int(iterations+1)):
            if iteration < self._start_iteration:
                continue

            success = False
            while not success:
                # time0 = time.time()
                '''
                Choose random NAVIGABLE point.
                If a path to the goal is already found, with 80% chance, we sample near that path.
                20% chance, explore elsewhere.
                '''
                sample_random = np.random.rand() < 0.2
                found_valid_new_node = False
                while not found_valid_new_node:
                    if sample_random:
                        rand_pt = PointHeading(self._pathfinder.get_random_navigable_point())
                        if abs(rand_pt.z - self._start.z) > 0.8:  # Must be on same plane as episode.
                            continue

                        # Shorten distance
                        closest_pt = self._closest_tree_pt(rand_pt)
                        rand_pt, has_changed = self._max_point(closest_pt, rand_pt)
                        if not has_changed or self._is_navigable(rand_pt):
                            found_valid_new_node = True
                    else:
                        if self._best_goal_node is None:
                            # sample_random = True; continue
                            best_path_pt = random.choice(self._shortest_path_points)
                        else:
                            best_path = self._get_path_to_start(self._best_goal_node)
                            best_path_pt = random.choice(best_path)

                        rand_r = 1.5 * np.sqrt(np.random.rand())  # TODO make this adjustable
                        rand_theta = np.random.rand() * 2 * np.pi
                        x = best_path_pt.x + rand_r * np.cos(rand_theta)
                        y = best_path_pt.y + rand_r * np.sin(rand_theta)
                        z = best_path_pt.z
                        rand_pt = PointHeading(self._pathfinder.snap_point([x, z, y]))  # MAY RETURN NAN NAN NAN

                        if not self._is_navigable(rand_pt):
                            continue

                        if self._best_goal_node is None:
                            closest_pt = self._closest_tree_pt(rand_pt)
                            rand_pt, has_changed = self._max_point(closest_pt, rand_pt)
                            if not has_changed or self._is_navigable(rand_pt):
                                found_valid_new_node = True
                        else:
                            found_valid_new_node = True
                # time1 = time.time()

                # Find valid neighbors
                nearby_nodes = []
                for pt in self._get_near_pts(rand_pt):
                    if (
                            self._euclid_2D(rand_pt, pt) < self._near_threshold  # within distance
                            and (rand_pt.x, rand_pt.y) != (pt.x, pt.y)  # not the same point again
                            and self._path_exists(pt, rand_pt)  # straight path exists
                    ):
                        nearby_nodes.append(pt)
                if not nearby_nodes:
                    continue
                # time2 = time.time()

                # Find best parent from valid neighbors
                min_cost = float('inf')
                for idx, pt in enumerate(nearby_nodes):
                    cost_from_parent, final_heading = self._cost_from_to(pt, rand_pt, return_heading=True)
                    new_cost = self._cost_from_start(pt) + cost_from_parent
                    if new_cost < min_cost:
                        min_cost = new_cost
                        best_final_heading = final_heading
                        best_parent_idx = idx
                        best_cost_from_parent = cost_from_parent
                # Sometimes there is just one nearby node whose new_cost is NaN. Continue if so.
                if min_cost == float('inf'):
                    continue
                # time3 = time.time()

                # Add+connect new node to graph
                rand_pt.heading = best_final_heading
                try:
                    self.tree[rand_pt] = nearby_nodes.pop(best_parent_idx)
                    self._cost_from_parent[rand_pt] = best_cost_from_parent
                    self.add_to_grid_hash(rand_pt)
                except IndexError:
                    continue

                # Rewire
                for pt in nearby_nodes:
                    if pt == self._start:
                        continue
                    cost_from_new_pt = self._cost_from_to(
                        rand_pt,
                        pt,
                        consider_end_heading=True
                    )
                    new_cost = self._cost_from_start(rand_pt) + cost_from_new_pt
                    if new_cost < self._cost_from_start(pt) and self._path_exists(rand_pt, pt):
                        self.tree[pt] = rand_pt
                        self._cost_from_parent[pt] = cost_from_new_pt

                # Update best path every so often
                if iteration % 50 == 0 or iteration % visualize_iterations == 0:
                    min_costs = []
                    for idx, pt in enumerate(self._get_near_pts(self._goal)):
                        if (
                                self._euclid_2D(pt, self._goal) < self._near_threshold
                                and self._path_exists(pt, self._goal)
                        ):
                            min_costs.append((
                                self._cost_from_start(pt) + self._cost_from_to(pt, self._goal),
                                idx,  # Tie-breaker for previous line when min is used
                                pt
                            ))
                    if len(min_costs) > 0:
                        self._best_goal_node = min(min_costs)[2]

                # Save tree and visualization to disk
                if (
                        iteration > 0
                        and iteration % visualize_iterations == 0
                        and self._directory is not None
                ):
                    img_path = os.path.join(self._vis_dir,
                                            '{}_{}.png'.format(iteration, os.path.basename(self._vis_dir)))
                    json_path = os.path.join(self._json_dir,
                                             '{}_{}.json'.format(iteration, os.path.basename(self._json_dir)))
                    self._visualize_tree(save_path=img_path, show=visualize_on_screen)
                    string_tree = self._string_tree()
                    with open(json_path, 'w') as f:
                        json.dump(string_tree, f)
                # self.times['time0'].append(time0)
                # self.times['time1'].append(time1)
                # self.times['time2'].append(time2)
                # self.times['time3'].append(time3)
                # self.times['time4'].append(time4)
                # self.times['time5'].append(time.time())
                # if len(self.times['time5']) == 50:
                #     for idx in range(50):
                #         avg1 = (self.times['time1'][idx]-self.times['time0'][idx])/(self.times['time5'][idx]-self.times['time0'][idx])
                #         avg2 = (self.times['time2'][idx]-self.times['time1'][idx])/(self.times['time5'][idx]-self.times['time0'][idx])
                #         avg3 = (self.times['time3'][idx]-self.times['time2'][idx])/(self.times['time5'][idx]-self.times['time0'][idx])
                #         avg4 = (self.times['time4'][idx]-self.times['time3'][idx])/(self.times['time5'][idx]-self.times['time0'][idx])
                #         avg5 = (self.times['time5'][idx]-self.times['time4'][idx])/(self.times['time5'][idx]-self.times['time0'][idx])
                #     print('{} 1:{} 2:{} 3:{} 4:{} 5:{}'.format(iteration, avg1, avg2, avg3, avg4, avg5))
                #     self.times = defaultdict(list)

                success = True

    def generate_topdown_img(self, meters_per_pixel=0.01):
        y = self._start.as_pos()[1]
        topdown = self._pathfinder.get_topdown_view(meters_per_pixel, y)
        topdown_bgr = np.zeros((*topdown.shape, 3), dtype=np.uint8)
        topdown_bgr[topdown == 0] = (255, 255, 255)
        topdown_bgr[topdown == 1] = (100, 100, 100)

        return topdown_bgr
