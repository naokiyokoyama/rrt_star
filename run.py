import argparse
import gzip
import habitat_sim
import json
import numpy as np
import quaternion as qt
from yacs.config import CfgNode as CN

# import yaml

from collections import defaultdict
from os import path as osp

from algo.utils import quat_to_rad

from algo.rrt_base import RRTStar

parser = argparse.ArgumentParser()
parser.add_argument("yaml_file", type=str, help="Path to .yaml file containing parameters")
parser.add_argument(
    "opts",
    default=None,
    nargs=argparse.REMAINDER,
    help="Modify config options from command line",
)
args = parser.parse_args()

params = CN()
params.set_new_allowed(True)
params.merge_from_file(args.yaml_file)

# Add config options to override if supplied
if args.opts:
    for k, v in zip(args.opts[0::2], args.opts[1::2]):
        assert k in params, f'{k} is not a valid parameter, cannot override.'
        params[k] = v

if params.PNG_FILE == '':
    # habitat_sim will be used: get all unique scene_ids contained in json.gz file
    with gzip.open(params.JSON_GZ, "r") as f:
        data = f.read()
    data = json.loads(data.decode("utf-8"))

    scene_eps = defaultdict(list)
    for ep in data["episodes"]:
        scene_eps[ep["scene_id"]].append(ep)

    for scene_id, episodes in scene_eps.items():
        # Generate simulator configuration
        glb_path = osp.join(params.SCENES_DIR, scene_id)
        assert osp.isfile(glb_path), f"{glb_path} does not exist"

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = glb_path

        # A camera is needed to enable the renderer that recomputes the navmesh...
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [habitat_sim.CameraSensorSpec()]

        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

        with habitat_sim.Simulator(cfg) as sim:

            # Compute navmesh
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            navmesh_settings.agent_height = 0.88
            navmesh_settings.agent_radius = 0.18

            sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

            for episode in episodes:
                # Skip if this is not the episode we are looking for
                if params.EPISODE_ID != -1 and episode['episode_id'] != params.EPISODE_ID:
                    continue

                # Get episode params
                start_position = episode["start_position"]
                start_quaternion = episode["start_rotation"]
                scene_name = episode["scene_id"]
                goal_position = episode["goals"][0]["position"]

                start_heading = quat_to_rad(qt.quaternion(*start_quaternion))

                rrt = RRTStar(
                    rrt_star=params.RRT_TYPE,
                    pathfinder=sim.pathfinder,
                    max_linear_velocity=params.MAX_LINEAR_VELOCITY,
                    max_angular_velocity=np.deg2rad(params.MAX_ANGULAR_VELOCITY),
                    near_threshold=params.NEAR_THRESHOLD,
                    max_distance=params.MAX_DISTANCE,
                    directory=params.OUT_DIR,
                )

                rrt.generate_tree(
                    start_position=start_position,
                    start_heading=start_heading,
                    goal_position=goal_position,
                    iterations=params.ITERATIONS,
                    visualize_on_screen=params.VISUALIZE_ON_SCREEN,
                    visualize_iterations=params.VISUALIZE_ITERATIONS,
                )

            sim.close()
else:
    # PNG file will be used
    assert osp.isfile(params.PNG_FILE), f"'{params.PNG_FILE}' does not exist!"

    start_position, goal_position = [
        np.array([params[i][0], 0.0, params[i][1]]) * params.METERS_PER_PIXEL
        for i in ['START_POSITION', 'GOAL_POSITION']
    ]
    rrt = RRTStar(
        rrt_star=params.RRT_TYPE,
        pathfinder=params.PNG_FILE,
        agent_radius=params.AGENT_RADIUS,
        meters_per_pixel=params.METERS_PER_PIXEL,
        max_linear_velocity=params.MAX_LINEAR_VELOCITY,
        max_angular_velocity=np.deg2rad(params.MAX_ANGULAR_VELOCITY),
        near_threshold=params.NEAR_THRESHOLD,
        max_distance=params.MAX_DISTANCE,
        directory=params.OUT_DIR,
    )

    rrt.generate_tree(
        start_position=start_position,
        start_heading=np.deg2rad(params.START_HEADING),
        goal_position=goal_position,
        iterations=params.ITERATIONS,
        visualize_on_screen=params.VISUALIZE_ON_SCREEN,
        visualize_iterations=params.VISUALIZE_ITERATIONS,
    )

