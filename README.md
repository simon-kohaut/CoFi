# CoFi — The Constitutional Filter

[![arXiv](https://img.shields.io/badge/arXiv-2412.18347-b31b1b.svg)](https://arxiv.org/abs/2412.18347)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FIROS60139.2025.11246215-blue.svg)](https://doi.org/10.1109/IROS60139.2025.11246215)
[![PyPI](https://img.shields.io/pypi/v/python-cofi)](https://pypi.org/project/python-cofi/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://github.com/simon-kohaut/CoFi/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://www.python.org/)

<p align="center">
  <a href="https://github.com/simon-kohaut/resin">Resin</a> reasoning ·
  <a href="https://github.com/HRI-EU/ProMis">ProMis</a> mission design ·
  <b>CoFi</b> tracking ·
  <a href="https://github.com/simon-kohaut/CoCo">CoCo</a> control
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/simon-kohaut/CoFi/main/docs/drones.gif" width="900" alt="Animation: CoFi tracks three drones through an obstacle course at 0.2, 0.5 and 1.0 m/s from noisy position fixes three times per second and the drones' accelerations in between. Its particles follow each drone and spread out between fixes, but stay off the obstacles that the drones' constitution forbids.">
</p>

<p align="center">
  <b>The Constitutional Filter: Bayesian Estimation of Compliant Agents</b><br>
  Simon Kohaut, Felix Divo, Benedict Flade, Devendra Singh Dhami, Julian Eggert, Kristian Kersting<br>
  <a href="https://arxiv.org/abs/2412.18347">Paper</a> · <a href="https://doi.org/10.1109/IROS60139.2025.11246215">IROS 2025</a> · <a href="#citation">Citation</a>
</p>

CoFi tracks agents that are expected to follow human-interpretable rules, their *constitution*.
The constitution is a [Resin](https://github.com/simon-kohaut/resin) program (see its [language reference](https://github.com/simon-kohaut/resin/blob/main/docs/language.md)) that [ProMis](https://github.com/HRI-EU/ProMis) evaluates over a map and facts about the agent, yielding the probability of compliance at every location.
CoFi folds this probability into a particle filter, weighted by how much the agent is trusted to comply, so the estimate stays where the agent is allowed to be, and falls back to a plain particle filter when the trust is zero.

## Installation

```bash
pip install python-cofi
```

To install from source:

```bash
git clone https://github.com/simon-kohaut/CoFi
cd CoFi
pip install -e .
```

This also installs [ProMis](https://hri-eu.github.io/ProMis/installation.html), which computes the landscapes of constitutions that CoFi tracks with.
Please refer to its usage guide for a reference on the environment representation and reasing aspects.

## Quick Start

### 1. Compute where the agent is allowed to be

```python
import time
import numpy as np
from promis import ProMis, StaRMap
from promis.geo import CartesianMap, CartesianPolygon, CartesianRasterBand, PolarLocation

# The constitution: stay off the obstacle
constitution = """
over(obstacle) <- source("/star_map/over/obstacle", Probability).
landscape if not over(obstacle).
landscape -> target("/landscape").
"""

# A 2 m x 1 m obstacle in a 10 m x 10 m area, whose position is known up to 10 cm
obstacle = CartesianPolygon.from_numpy(np.array([[-1, 1, 1, -1], [-0.5, -0.5, 0.5, 0.5]]), location_type="obstacle")
area = CartesianMap(PolarLocation(0, 0), {"obstacle": [obstacle]})
area.apply_covariance({"obstacle": 0.1**2 * np.eye(2)})

# The probability of complying with the constitution across the area
points = CartesianRasterBand(PolarLocation(0, 0), (50, 50), 10, 10)
star_map = StaRMap(area)
star_map.initialize(points, number_of_random_maps=25, logic=constitution)
promis = ProMis(star_map, constitution, 50 * 50)
promis.initialize(points)
time.sleep(0.05)  # Resin's writers deliver values asynchronously
```

The constitution has to publish the probability of compliance on the `/landscape` topic, as in `landscape -> target("/landscape").`: this is the landscape ProMis hands to CoFi.

### 2. Track the agent

```python
from cofi import ConstitutionalFilter

# An agent passing above the obstacle at 1 m/s, observed every second with 20 cm of noise
timestamps = np.arange(10.0)
positions = np.stack([timestamps - 4.5, np.full(10, 1.0)]).T + np.random.default_rng(0).normal(0, 0.2, (10, 2))
positions[4:7] = np.nan  # no observations for three seconds

cofi = ConstitutionalFilter(
    promis, trust=0.9,
    sensor_std=0.2, process_noise=0.5, initial_position_std=0.2, initial_velocity_std=1.0,
)
result = cofi.run(positions, timestamps, seed=0)

result.estimates       # (T, 4): east, north, v_east, v_north
result.position_error  # (T,): distance to the observed positions, NaN where there is none
```

With `trust=0.0`, `ConstitutionalFilter` is a plain particle filter.
Rows of `positions` that are NaN mark missing observations: the particles are then only predicted and weighted by the constitution, which keeps the estimate where the agent is allowed to be until it is observed again.
`sensor_std` is the observation noise per second since the last observation, and `process_noise` the change of velocity per second, both in the units of the positions.
With `run(..., accelerations=...)`, measured accelerations, e.g., from an IMU, move the particles' velocities between observations, and `process_noise` is their noise.
When the filter steps more often than the agent is observed, `constitution_interval` sets the seconds over which the constitution weighs in once, so that its influence does not depend on the step rate.

### 3. Track the agent online

When observations arrive one at a time, e.g., while ProMis updates the landscape, step the filter online instead:

```python
cofi.start(positions[0], timestamps[0], seed=0)
for position, timestamp in zip(positions, timestamps):
    estimate = cofi.step(position, timestamp)  # east, north, v_east, v_north
```

## Usage Example

We provide a usage example, [examples/drones.py](https://github.com/simon-kohaut/CoFi/blob/main/examples/drones.py), that tracks drones flying through the obstacle course below with a known set of rules.
The flights were recorded with [CoCo](https://github.com/simon-kohaut/CoCo) and are included in [examples/data](https://github.com/simon-kohaut/CoFi/tree/main/examples/data).

<p align="center">
  <img src="https://raw.githubusercontent.com/simon-kohaut/CoFi/main/examples/data/linearized_image.png" width="500" alt="Top-down view of the obstacle course: red, yellow, green and blue blocks on a dark floor.">
</p>

The drones follow this Resin program, with distances in meters:

```prolog
over(red) <- source("/star_map/over/red", Probability).
over(green) <- source("/star_map/over/green", Probability).
distance(yellow) <- source("/star_map/distance/yellow", Density).
velocity <- source("/velocity", Number).

rule_1 if not over(red).
rule_2 if distance(yellow) > 0.05.
rule_3 if not over(green).
rule_3 if over(green) and velocity < 0.8.

landscape if rule_1 and rule_2 and rule_3.
landscape -> target("/landscape").
```

To render the animation above, simply run:

```bash
pip install -e ".[examples]"
python examples/drones.py docs/drones.gif
```

The extra dependencies include ffmpeg on Windows, macOS and Linux; on other platforms, install ffmpeg separately.

## Citation

If you use CoFi, please cite the paper below. CoFi builds on [ProMis](https://github.com/HRI-EU/ProMis); please also consider citing its [publications](https://github.com/HRI-EU/ProMis#cite).

```bibtex
@inproceedings{kohaut2025cofi,
  title     = {The Constitutional Filter: Bayesian Estimation of Compliant Agents},
  author    = {Kohaut, Simon and Divo, Felix and Flade, Benedict and Dhami, Devendra Singh
               and Eggert, Julian and Kersting, Kristian},
  booktitle = {Proceedings of the IEEE/RSJ International Conference on Intelligent
               Robots and Systems (IROS)},
  year      = {2025},
  doi       = {10.1109/IROS60139.2025.11246215}
}
```
