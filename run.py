import argparse
import gzip
import habitat_sim
import json
import numpy as np
import quaternion as qt

from collections import defaultdict
from os import path as osp

from algo.rrt_base import RRTStar
from algo.utils import quat_to_rad

parser = argparse.ArgumentParser()
parser.add_argument("json_gz", type=str)
parser.add_argument("dataset_dir", type=str)
parser.add_argument("out_dir", type=str)
parser.add_argument("--max_linear_velocity", type=float, default=0.25)  # in m/s
parser.add_argument(
    "--max_angular_velocity", type=float, default=np.pi / 180 * 10
)  # in rad/s
parser.add_argument("--goal_minimum_distance", type=float, default=0.2)  # in m
parser.add_argument("--near_threshold", type=float, default=1.5)  # in m
parser.add_argument("--max_distance", type=float, default=1.5)  # in m
parser.add_argument("--visualize_on_screen", action="store_true")
parser.add_argument("--iterations", type=int, default=5e3)
parser.add_argument("--visualize_iterations", type=int, default=500)
args = parser.parse_args()

# Get all unique scene_ids contained in json.gz file
with gzip.open(args.json_gz, "r") as f:
    data = f.read()
data = json.loads(data.decode("utf-8"))

scene_eps = defaultdict(list)
for ep in data["episodes"]:
    scene_eps[ep["scene_id"]].append(ep)

for scene_id, episodes in scene_eps.items():
    # Generate simulator configuration
    glb_path = osp.join(args.dataset_dir, scene_id)
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

            # Get episode params
            start_position = episode["start_position"]
            start_quaternion = episode["start_rotation"]
            scene_name = episode["scene_id"]
            goal_position = episode["goals"][0]["position"]

            start_heading = quat_to_rad(qt.quaternion(*start_quaternion))

            rrt = RRTStar(
                # 'unicycle',
                'pointturn',
                pathfinder=sim.pathfinder,
                max_linear_velocity=args.max_linear_velocity,
                max_angular_velocity=args.max_angular_velocity,
                near_threshold=args.near_threshold,
                max_distance=args.max_distance,
                directory=args.out_dir,
            )

            rrt.generate_tree(
                start_position=start_position,
                start_heading=start_heading,
                goal_position=goal_position,
                iterations=args.iterations,
                visualize_on_screen=args.visualize_on_screen,
                visualize_iterations=args.visualize_iterations,
            )

        sim.close()
