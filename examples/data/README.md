# Drone Flights

These files are taken from the [Constitutional Controller (CoCo)](https://github.com/simon-kohaut/CoCo), which steered drones through an obstacle course by a known constitution.

* `flights/{speed}/*.csv`: the flights at 0.2, 0.5 and 1.0 m/s, with the time in nanoseconds and the position in meters, as recorded with a Lighthouse positioning system. Only the columns used by [../drones.py](../drones.py) are kept; see CoCo for the full recordings.
* `obstacles.csv`: the obstacle blocks detected by CoCo's vision module, as their color and corners in pixels of the scene image.
* `scene.png`: the linearized top-down image of the obstacle course, in grayscale.
* `linearized_image.png`: the same image in color.
