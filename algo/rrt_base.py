from os import path as osp
import os
import glob
import json
import numpy as np
import random
import tqdm
from collections import defaultdict

from .utils import PointHeading


# class RRTStarSim(RRTStarBase):
#     def __init__(
#         self,
#         pathfinder,
#         max_linear_velocity,
#         max_angular_velocity,
#         near_threshold,
#         max_distance,
#         directory=None,
#     ):
#         super().__init__(
#             max_linear_velocity=max_linear_velocity,
#             max_angular_velocity=max_angular_velocity,
#             near_threshold=near_threshold,
#             max_distance=max_distance,
#             directory=directory,
#         )
#
#         self._pathfinder = pathfinder


class RRTStarBase:
    def __init__(
        self,
        max_linear_velocity,
        max_angular_velocity,
        near_threshold,
        max_distance,
        directory=None,
    ):
        assert near_threshold >= max_distance, (
            f"near_threshold ({near_threshold}) must be greater than or"
            f"equal to max_distance ({max_distance})"
        )
        self._near_threshold = near_threshold
        self._max_distance = max_distance
        self._max_linear_velocity = max_linear_velocity
        self._max_angular_velocity = max_angular_velocity
        self._directory = directory

        if self._directory is not None:
            self._vis_dir = osp.join(self._directory, "visualizations")
            self._json_dir = osp.join(self._directory, "tree_jsons")
            for i in [self._vis_dir, self._json_dir]:
                if not osp.isdir(i):
                    os.makedirs(i)

        # TODO: Search the tree with binary search
        self.tree = {}
        self.grid_hash = defaultdict(list)
        self._best_goal_node = None
        self._top_down_img = None
        self._start_iteration = 0
        self._start = None
        self._goal = None
        self._cost_from_parent = {}

        self.x_min, self.y_min = None, None

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

    def _euclid_2D(self, p1, p2):
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

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

    def _load_tree_from_json(self, json_path=None):
        """
        Attempts to recover the latest existing tree from the json directory,
        or provided json_path.
        """
        if json_path is None:
            existing_jsons = glob.glob(osp.join(self._json_dir, "*.json"))
            if len(existing_jsons) > 0:
                get_num_iterations = lambda x: int(osp.basename(x).split("_")[0])
                json_path = sorted(existing_jsons, key=get_num_iterations)[-1]
            else:
                return None

        with open(json_path) as f:
            latest_string_tree = json.load(f)

        start_str = latest_string_tree["start"]
        if self._start is None:
            self._start = self._str_to_pt(start_str)

        goal_str = latest_string_tree["goal"]
        if self._goal is None:
            self._goal = self._str_to_pt(goal_str)

        if self.x_min is None:
            self._set_offsets()

        for k, v in latest_string_tree["graph"].items():
            pt = self._str_to_pt(k)
            if k == latest_string_tree["best_goal_node"]:
                self._best_goal_node = pt

            if v == "":  # start node is key
                continue
            if v == start_str:
                self.tree[pt] = self._start
                self.add_to_grid_hash(pt)
            else:
                pt_v = self._str_to_pt(v)
                self.tree[pt] = pt_v
                self.add_to_grid_hash(pt)

        self._start_iteration = int(osp.basename(json_path).split("_")[0]) + 1

        return None

    def _str_to_pt(self, pt_str):
        x, y, z, heading = pt_str.split("_")
        point = (float(x), float(z), float(y))
        return PointHeading(point, float(heading))

    def _get_best_path(self):
        if self._best_goal_node is None:
            return []
        return self._get_path_to_start(self._best_goal_node) + [self._goal]

    def add_to_grid_hash(self, pt):
        i = int((pt.x - self.x_min) // self._near_threshold)
        j = int((pt.y - self.y_min) // self._near_threshold)
        self.grid_hash[(i, j)].append(pt)

    def _set_offsets(self):
        self.x_min, self.y_min = float("inf"), float("inf")
        for v in self._pathfinder.build_navmesh_vertices():
            pt = PointHeading(v)
            # Make sure it's on the same elevation as the start point
            if abs(pt.z - self._start.z) < 0.8:
                self.x_min = min(self.x_min, pt.x)
                self.y_min = min(self.y_min, pt.y)

    def _string_tree(self):
        """
        Return the current graph as a dictionary comprised of strings to save
        to disk.
        """
        string_graph = {}
        for k, v in self.tree.items():
            if k == self._start:
                string_graph[k._str_key()] = ""
            else:
                string_graph[k._str_key()] = v._str_key()

        string_tree = {
            "start": self._start._str_key(),
            "goal": self._goal._str_key(),
        }

        if self._best_goal_node is not None:
            string_tree["best_goal_node"] = self._best_goal_node._str_key()
            string_tree["best_path_time"] = self._cost_from_start(
                self._best_goal_node
            ) + self._cost_from_to(self._best_goal_node, self._goal)
        else:
            string_tree["best_goal_node"] = ""
            string_tree["best_path_time"] = -1

        # Add the best path
        string_tree["best_path_raw"] = [i._str_key() for i in self._get_best_path()]
        string_tree["graph"] = string_graph

        return string_tree

    def generate_topdown_img(self, meters_per_pixel=0.01):
        y = self._start.as_pos()[1]
        topdown = self._pathfinder.get_topdown_view(meters_per_pixel, y)
        topdown_bgr = np.zeros((*topdown.shape, 3), dtype=np.uint8)
        topdown_bgr[topdown == 0] = (255, 255, 255)
        topdown_bgr[topdown == 1] = (100, 100, 100)

        return topdown_bgr

    def generate_tree(
        self,
        start_position,
        start_heading,
        goal_position,
        json_path=None,
        iterations=5e4,
        visualize_on_screen=False,
        visualize_iterations=500,
        seed=0,
    ):
        """
        This is the main algorithm that produces the tree rooted at the starting pose.

        :param start_position:
        :param start_heading:
        :param goal_position:
        :param json_path:
        :param iterations:
        :param visualize_on_screen:
        :param visualize_iterations:
        :param seed:
        :return:
        """
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
                """
                Choose random NAVIGABLE point.
                If a path to the goal is already found, with 80% chance, we sample near that path.
                20% chance, explore elsewhere.
                """
                sample_random = np.random.rand() < 0.2
                found_valid_new_node = False
                while not found_valid_new_node:
                    if sample_random:
                        rand_pt = PointHeading(
                            self._pathfinder.get_random_navigable_point()
                        )
                        if (
                            abs(rand_pt.z - self._start.z) > 0.8
                        ):  # Must be on same plane as episode.
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

                        rand_r = 1.5 * np.sqrt(
                            np.random.rand()
                        )  # TODO make this adjustable
                        rand_theta = np.random.rand() * 2 * np.pi
                        x = best_path_pt.x + rand_r * np.cos(rand_theta)
                        y = best_path_pt.y + rand_r * np.sin(rand_theta)
                        z = best_path_pt.z
                        rand_pt = PointHeading(
                            self._pathfinder.snap_point([x, z, y])
                        )  # MAY RETURN NAN NAN NAN

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
                        self._euclid_2D(rand_pt, pt)
                        < self._near_threshold  # within distance
                        and (rand_pt.x, rand_pt.y)
                        != (pt.x, pt.y)  # not the same point again
                        and self._path_exists(pt, rand_pt)  # straight path exists
                    ):
                        nearby_nodes.append(pt)
                if not nearby_nodes:
                    continue
                # time2 = time.time()

                # Find best parent from valid neighbors
                min_cost = float("inf")
                for idx, pt in enumerate(nearby_nodes):
                    cost_from_parent, final_heading = self._cost_from_to(
                        pt, rand_pt, return_heading=True
                    )
                    new_cost = self._cost_from_start(pt) + cost_from_parent
                    if new_cost < min_cost:
                        min_cost = new_cost
                        best_final_heading = final_heading
                        best_parent_idx = idx
                        best_cost_from_parent = cost_from_parent
                # Sometimes there is just one nearby node whose new_cost is NaN. Continue if so.
                if min_cost == float("inf"):
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
                        rand_pt, pt, consider_end_heading=True
                    )
                    new_cost = self._cost_from_start(rand_pt) + cost_from_new_pt
                    if new_cost < self._cost_from_start(pt) and self._path_exists(
                        rand_pt, pt
                    ):
                        self.tree[pt] = rand_pt
                        self._cost_from_parent[pt] = cost_from_new_pt

                # Update best path every so often
                if iteration % 50 == 0 or iteration % visualize_iterations == 0:
                    min_costs = []
                    for idx, pt in enumerate(self._get_near_pts(self._goal)):
                        if self._euclid_2D(
                            pt, self._goal
                        ) < self._near_threshold and self._path_exists(pt, self._goal):
                            min_costs.append(
                                (
                                    self._cost_from_start(pt)
                                    + self._cost_from_to(pt, self._goal),
                                    idx,  # Tie-breaker for previous line when min is used
                                    pt,
                                )
                            )
                    if len(min_costs) > 0:
                        self._best_goal_node = min(min_costs)[2]

                # Save tree and visualization to disk
                if (
                    iteration > 0
                    and iteration % visualize_iterations == 0
                    and self._directory is not None
                ):
                    img_path = osp.join(
                        self._vis_dir,
                        "{}_{}.png".format(iteration, osp.basename(self._vis_dir)),
                    )
                    json_path = osp.join(
                        self._json_dir,
                        "{}_{}.json".format(iteration, osp.basename(self._json_dir)),
                    )
                    self._visualize_tree(save_path=img_path, show=visualize_on_screen)
                    string_tree = self._string_tree()
                    with open(json_path, "w") as f:
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
