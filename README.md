# Kinodynamic RRT-Star

This repo approximates the fastest completion time from a starting pose to a goal position based on the dynamics of the
agent. Details are [here](https://arxiv.org/abs/2103.08022).

Currently, this repo supports the following dynamics:

* Unicycle
* Pointturn

## Installation

You must have habitat_sim installed to use this repo. Repo with installation instructions is located 
[here](https://github.com/facebookresearch/habitat-sim/). The use of conda is strongly suggested.

You will also need to install OpenCV and yacs:
```bash
pip install opencv-python yacs
```

## Running the scripts

This repo can accept either the `.json.gz` files that [habitat-lab](https://github.com/facebookresearch/habitat-lab/) uses
for training PointNav agents, or `.png` files that represent a 2D floor plan. For the `.png` files, white pixels 
represent navigable locations and black/gray one represent walls or obstacles (see `maps/example_map.png` for reference).

You must use a `.yaml` file to configure the various parameters needed for running the script (see files in `configs` 
folder). 

Here is example of proper usage:

```bash
python run.py configs/png.yaml
```
 