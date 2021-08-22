from os import path as osp
import os
import cv2
import glob
import json
import numpy as np
import random
import tqdm
from collections import defaultdict

from .rrt_unicycle import RRTStarUnicycleSelect
from .rrt_pointturn import RRTStarPTSelect
from .utils import PointHeading

from habitat_sim.nav import ShortestPath

RRTStarMapping = {
    "unicycle": RRTStarUnicycleSelect,
    "pointturn": RRTStarPTSelect,
}


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

    def _max_point(self, p1, p2):
        euclid_dist = self._euclid_2D(p1, p2)
        if euclid_dist <= self._max_distance:
            return p2, False

        new_x = p1.x + (p2.x - p1.x * self._max_distance / euclid_dist)
        new_y = p1.y + (p2.y - p1.y * self._max_distance / euclid_dist)
        new_z = p1.z
        p_new = PointHeading((new_x, new_z, new_y))  # MAY RETURN non navigable point

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

    def _cost_from_start(self, pt):
        path = self._get_path_to_start(pt)
        cost = 0
        for parent, child in zip(path[:-1], path[1:]):
            if child not in self._cost_from_parent:
                self._cost_from_parent[child] = self._cost_from_to(
                    parent, child, consider_end_heading=True
                )
            cost += self._cost_from_parent[child]
        return cost

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

    def make_path_finer(self, path, precision=2, resolution=0.01):
        all_pts = []
        for pt, new_pt in zip(path[:-1], path[1:]):
            all_pts.append(pt)
            all_pts += self._get_intermediate_pts(
                pt, new_pt, precision=precision, resolution=resolution
            )
            all_pts.append(new_pt)

        return all_pts

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
                    if (
                        sample_random
                        or not self._shortest_path_points
                        and self._best_goal_node is None
                    ):
                        rand_pt = self._sample_random_point()
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
                            self._snap_point([x, z, y])
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
                        "{}_{}.png".format(iteration, osp.basename(self._directory)),
                    )
                    json_path = osp.join(
                        self._json_dir,
                        "{}_{}.json".format(iteration, osp.basename(self._directory)),
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

    """
    Visualization methods
    """

    def generate_topdown_img(self, meters_per_pixel=0.01):
        y = self._start.as_pos()[1]
        topdown = self._pathfinder.get_topdown_view(meters_per_pixel, y)
        topdown_bgr = np.zeros((*topdown.shape, 3), dtype=np.uint8)
        topdown_bgr[topdown == 0] = (255, 255, 255)
        topdown_bgr[topdown == 1] = (100, 100, 100)

        return topdown_bgr

    def _draw_all_edges(self, img):
        for node, node_parent in self.tree.items():
            if node_parent is None:  # Start point has no parent
                continue
            fine_path = (
                [node_parent]
                + self._get_intermediate_pts(node_parent, node, resolution=0.01)
                + [node]
            )
            for pt, next_pt in zip(fine_path[:-1], fine_path[1:]):
                cv2.line(
                    img,
                    (self._scale_x(pt.x), self._scale_y(pt.y)),
                    (self._scale_x(next_pt.x), self._scale_y(next_pt.y)),
                    (0, 102, 255),
                    1,
                )

        return img

    def _draw_best_path(self, img, path):
        if path is not None:
            fine_path = self.make_path_finer(path)
        else:
            fine_path = self.make_path_finer(self._get_best_path())
        for pt, next_pt in zip(fine_path[:-1], fine_path[1:]):
            cv2.line(
                img,
                (self._scale_x(pt.x), self._scale_y(pt.y)),
                (self._scale_x(next_pt.x), self._scale_y(next_pt.y)),
                (0, 255, 0),
                3,
            )

        return img

    def _draw_start_pose(self, img):
        start_x, start_y = self._scale_x(self._start.x), self._scale_y(self._start.y)
        cv2.circle(img, (start_x, start_y), 8, (255, 192, 15), -1)
        LINE_SIZE = 10
        heading_end_pt = (
            int(start_x + LINE_SIZE * np.cos(self._start.heading)),
            int(start_y + LINE_SIZE * np.sin(self._start.heading)),
        )
        cv2.line(img, (start_x, start_y), heading_end_pt, (0, 0, 0), 3)

        return img

    def _draw_goal_point(self, img):
        SQUARE_SIZE = 6
        cv2.rectangle(
            img,
            (
                self._scale_x(self._goal.x) - SQUARE_SIZE,
                self._scale_y(self._goal.y) - SQUARE_SIZE,
            ),
            (
                self._scale_x(self._goal.x) + SQUARE_SIZE,
                self._scale_y(self._goal.y) + SQUARE_SIZE,
            ),
            (0, 0, 255),
            -1,
        )

        return img

    def _draw_shortest_path_waypoints(self, img):
        if self._shortest_path_points is None:
            self._get_shortest_path_points()
        for i in self._shortest_path_points:
            cv2.circle(
                img,
                (self._scale_x(i.x), self._scale_y(i.y)),
                3,
                (255, 192, 15),
                -1,
            )
        return img

    def _draw_fastest_path_waypoints(self, img, path):
        if path is None:
            path = self._get_best_path()[1:-1]
        for i in path:
            cv2.circle(
                img,
                (self._scale_x(i.x), self._scale_y(i.y)),
                3,
                (0, 0, 255),
                -1,
            )
            LINE_SIZE = 8
            heading_end_pt = (
                int(self._scale_x(i.x) + LINE_SIZE * np.cos(i.heading)),
                int(self._scale_y(i.y) + LINE_SIZE * np.sin(i.heading)),
            )
            cv2.line(
                img,
                (self._scale_x(i.x), self._scale_y(i.y)),
                heading_end_pt,
                (0, 0, 0),
                1,
            )

        return img

    def _generate_top_down(self, meters_per_pixel):
        self._top_down_img = self.generate_topdown_img(
            meters_per_pixel=meters_per_pixel
        )

        # Crop image to just valid points
        mask = cv2.cvtColor(self._top_down_img, cv2.COLOR_BGR2GRAY)
        mask[mask == 255] = 0
        x, y, w, h = cv2.boundingRect(mask)
        self._top_down_img = self._top_down_img[y : y + h, x : x + w]

        # Determine scaling needed
        self._scale_x = lambda x: int((x - self.x_min) / meters_per_pixel)
        self._scale_y = lambda y: int((y - self.y_min) / meters_per_pixel)

    def _visualize_tree(
        self,
        meters_per_pixel=0.01,
        show=False,
        path=None,  # Option to visualize another path
        draw_all_edges=True,
        save_path=None,
    ):
        """
        Save and/or visualize the current tree and the best path found so far
        """
        if self._top_down_img is None:
            self._generate_top_down(meters_per_pixel)

        top_down_img = self._top_down_img.copy()

        if draw_all_edges:
            top_down_img = self._draw_all_edges(top_down_img)
        top_down_img = self._draw_best_path(top_down_img, path)
        top_down_img = self._draw_start_pose(top_down_img)
        top_down_img = self._draw_goal_point(top_down_img)
        top_down_img = self._draw_shortest_path_waypoints(top_down_img)
        top_down_img = self._draw_fastest_path_waypoints(top_down_img, path)

        # Display image on screen
        if show:
            cv2.imshow("top_down_img", top_down_img)
            cv2.waitKey(1)

        # Save the visualization to disk
        if save_path is not None:
            cv2.imwrite(save_path, top_down_img)

        return top_down_img


class RRTStarSim(RRTStarBase):
    def __init__(
        self,
        pathfinder,
        max_linear_velocity,
        max_angular_velocity,
        near_threshold,
        max_distance,
        directory=None,
    ):
        super().__init__(
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            near_threshold=near_threshold,
            max_distance=max_distance,
            directory=directory,
        )

        self._pathfinder = pathfinder
        self.pathfinder_type = "habitat_sim"

    def _is_navigable(self, pt, max_y_delta=0.5):
        return self._pathfinder.is_navigable(pt.as_pos(), max_y_delta=max_y_delta)

    def _get_shortest_path_points(self):
        sp = ShortestPath()
        sp.requested_start = self._start.as_pos()
        sp.requested_end = self._goal.as_pos()
        self._pathfinder.find_path(sp)
        self._shortest_path_points = [PointHeading(i) for i in sp.points]

    def _max_point(self, p1, p2):
        p_new, moved = super()._max_point(p1, p2)
        if not moved:
            return p_new, moved

        new_x, new_y, new_z = p_new.as_pos()
        p_new = PointHeading(
            self._pathfinder.snap_point([new_x, new_z, new_y])
        )  # MAY RETURN NAN NAN NAN

        return p_new, True

    def _sample_random_point(self):
        return PointHeading(self._pathfinder.get_random_navigable_point())

    def _snap_point(self, xzy):
        return self._pathfinder.snap_point(xzy)


class RRTStarPNG(RRTStarBase):
    def __init__(
        self,
        pathfinder,
        max_linear_velocity,
        max_angular_velocity,
        near_threshold,
        max_distance,
        directory=None,
    ):
        super().__init__(
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            near_threshold=near_threshold,
            max_distance=max_distance,
            directory=directory,
        )

        img = cv2.imread(pathfinder, cv2.IMREAD_UNCHANGED)
        self._map = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        self._map[self._map > 240] = 255
        self._map[self._map <= 240] = 0
        blur_radius = int(round(agent_radius / meters_per_pixel))
        self._map = cv2.blur(self._map, (blur_radius, blur_radius))
        self._map_height = float(img.shape[0]) * meters_per_pixel
        self._map_width = float(img.shape[1]) * meters_per_pixel
        self._top_down_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self._scale_x = lambda x: int(x / meters_per_pixel)
        self._scale_y = lambda y: int((y) / meters_per_pixel)

        self.x_min = 0
        self.y_min = 0

        self.pathfinder_type = "png"

    def _is_navigable(self, pt):
        px = self._scale_x(pt.x)
        py = self._scale_x(pt.y)

        try:
            return self._map[py, px] == 255
        except IndexError:
            return False

    def _sample_random_point(self):
        x_rand = np.random.rand() * self._map_width
        y_rand = np.random.rand() * self._map_height
        return PointHeading((x_rand, 0, y_rand))

    def _snap_point(self, xzy):
        # No z-snapping required for flat PNG maps
        return xzy

    def _get_shortest_path_points(self):
        # No shortest path can be found for PNG maps
        self._shortest_path_points = []


def RRTStar(rrt_star, pathfinder, *args, **kwargs):
    parent_class = RRTStarSim if type(pathfinder) != str else RRTStarPNG
    rrt_class = RRTStarMapping[rrt_star](parent_class)

    return rrt_class(pathfinder=pathfinder, *args, **kwargs)
