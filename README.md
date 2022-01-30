# Kinodynamic RRT*

![rrt_star_apartment](https://raw.githubusercontent.com/naokiyokoyama/website_media/master/imgs/sct/apartment_short_vs_fast.png)

_The shortest path is not always fastest, depending on the agent's dynamics (left). RRT*-Unicycle explicitly considers the agent's dynamics to measure how well its learned behavior approximates the fastest possible one (right)_

This repo approximates the fastest completion time from a starting pose to a goal position based on the dynamics of the
agent. Details are [here](https://arxiv.org/abs/2103.08022).

Currently, this repo supports the following dynamics:

* Unicycle
* Pointturn

## Citing
If you use this code in your research, please cite the following [paper](https://arxiv.org/abs/2103.08022):

```
@inproceedings{sct21iros,
  title     =     {Success Weighted by Completion Time: {A} Dynamics-Aware Evaluation Criteria for Embodied Navigation},
  author    =     {Naoki Yokoyama and Sehoon Ha and Dhruv Batra},
  booktitle =     {2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      =     {2021}
}
```

## Installation

If you want to use the unicycle dynamics model or use a navmesh (vs. a PNG file), you need to install habitat-sim. If you just want to use the PNG method and the point-turn model, you don't need to install it. Habitat-sim repo with installation instructions is located 
[here](https://github.com/facebookresearch/habitat-sim/). The use of conda is strongly suggested.

You will need to install the following packages:
```bash
pip install numpy numpy-quaternion yacs opencv-python tqdm
```

## Running the scripts

This repo can accept either the `.json.gz` files that [habitat-lab](https://github.com/facebookresearch/habitat-lab/) uses
for training PointNav agents, or `.png` files that represent a 2D floor plan. For the `.png` files, white pixels 
represent navigable locations and black/gray ones represent walls or obstacles (see `maps/example_map.png` for reference).

You must use a `.yaml` file to configure the various parameters needed for running the script.

**Please read `.yaml` files in `configs` folder for list of configurable parameters and descriptions for each one.**

After the yaml file has been properly configured, simply pass it to the `run.py` script. Here is an example of proper usage:

```bash
python run.py configs/png.yaml
```
 
