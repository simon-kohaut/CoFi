"""Usage example: tracks drones that follow a known set of rules with CoFi.

The flights in examples/data were recorded with CoCo (https://github.com/simon-kohaut/CoCo).
Run from the CoFi root:

    python examples/drones.py docs/drones.gif
"""

#
# Copyright (c) Simon Kohaut, Felix Divo, and contributors
#
# This file is part of CoFi and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with CoFi.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from time import monotonic, sleep

# Third Party
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
# The ffmpeg binary bundled with imageio-ffmpeg, or one on the PATH where none is bundled
matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import to_rgba
from PIL import Image
from scipy.interpolate import CubicSpline
from promis import ProMis, StaRMap
from promis.geo import CartesianMap, CartesianPolygon, CartesianRasterBand, PolarLocation

# CoFi
from cofi import ConstitutionalFilter

#: The constitution the drones were steered by in CoCo, in meters: stay off red blocks,
#: keep 5 cm from yellow blocks, and only cross green blocks slower than 0.8 m/s.
DRONE_CONSTITUTION = r"""
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
"""

#: The area of the landscape in meters, centered on the obstacle course and enclosing all
#: flights; beyond it, CoFi interpolates the landscape from its nearest point.
AREA_SIZE = (5.0, 4.0)
LANDSCAPE_RESOLUTION = (100, 80)

#: The speeds the drones were flown at in m/s, one panel each.
SPEEDS = (0.2, 0.5, 1.0)

#: The rate of the flight logs in Hz.
LOG_RATE = 10

#: The parameters of the filter, besides its sensor and process noise, which follow the tracker
#: and the IMU.
PARAMETERS = dict(
    trust=1.0,  # The drones follow their constitution exactly
    number_of_particles=2_000,
    initial_position_std=0.15,
    initial_velocity_std=0.2,
)

# The color of CoCo in its figures
COLOR = "#2ecc71"


def load_scene(data: Path) -> tuple[CartesianMap, Image.Image, tuple[float, float]]:
    """The obstacle map detected by CoCo's vision module, the scene image, and its size in meters."""

    image = Image.open(data / "scene.png")
    boxes = pd.read_csv(data / "obstacles.csv")

    # The obstacle blocks measure 0.8 x 0.2 m, which sets the scale of the linearized image
    pixel_width = 0.8 / ((277 + 281 + 245) / 3)
    pixel_height = 0.2 / ((65 + 65 + 74) / 3)
    width, height = image.width * pixel_width, image.height * pixel_height

    features = defaultdict(list)
    for _, box in boxes.iterrows():
        corners = np.array(json.loads(box["poly_oriented"].replace("(", "[").replace(")", "]")), dtype=float)
        if corners.shape[0] != 4:
            continue

        # Pixels to meters, with the origin at the center of the cage and y pointing up
        corners[:, 0] = corners[:, 0] * pixel_width - width / 2
        corners[:, 1] = height / 2 - corners[:, 1] * pixel_height
        features[box["color"]].append(CartesianPolygon.from_numpy(corners.T, location_type=box["color"]))

    scene = CartesianMap(PolarLocation(0, 0), features)
    scene.apply_covariance({location_type: 0.05**2 * np.eye(2) for location_type in features})  # 5 cm uncertainty

    return scene, image, (width, height)


def make_promis(scene: CartesianMap, speed: float) -> ProMis:
    """ProMis computing the probability of complying with the constitution when flying at `speed`."""

    points = CartesianRasterBand(PolarLocation(0, 0), LANDSCAPE_RESOLUTION, *AREA_SIZE)
    np.random.seed(2024)  # ProMis samples the random maps from NumPy's global state
    star_map = StaRMap(scene)
    star_map.initialize(points, number_of_random_maps=25, logic=DRONE_CONSTITUTION)

    dimension = LANDSCAPE_RESOLUTION[0] * LANDSCAPE_RESOLUTION[1]
    promis = ProMis(star_map, DRONE_CONSTITUTION, dimension)
    promis.initialize(points)
    promis.make_writer_for("velocity").write(speed * np.ones(dimension), monotonic())
    sleep(0.2)  # writers deliver values asynchronously

    return promis


def load_flight(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Timestamps in seconds and Lighthouse positions in meters of the airborne part of a flight.

    The positions are estimated at the rate of the log, while the logged timestamps jitter with
    the radio link, so the timestamps follow the nominal rate.
    """

    flight = pd.read_csv(path)
    flight = flight[flight["stateEstimateZ"] > 0.2]
    positions = flight[["stateEstimateX", "stateEstimateY"]].to_numpy()
    timestamps = np.arange(len(positions)) / LOG_RATE

    return timestamps, positions


def typical_flight(data: Path, speed: float) -> Path:
    """The flight at `speed` with the median duration."""

    paths = sorted((data / "flights" / f"{speed}").glob("*.csv"), key=lambda path: int(path.stem))
    durations = [load_flight(path)[0][-1] for path in paths]
    return paths[np.argsort(durations)[len(paths) // 2]]


def interpolate(
    timestamps: np.ndarray, positions: np.ndarray, rate: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Timestamps, positions and accelerations along a smooth trajectory through the positions at `rate`.

    The accelerations are differentiated on the rate of the positions and only then interpolated,
    since the second derivative of a spline through every position is dominated by their noise.
    """

    accelerations = np.gradient(np.gradient(positions, timestamps, axis=0), timestamps, axis=0)
    interpolated = np.arange(0, timestamps[-1], 1 / rate)
    return (
        interpolated,
        CubicSpline(timestamps, positions, axis=0)(interpolated),
        np.column_stack([np.interp(interpolated, timestamps, axis) for axis in accelerations.T]),
    )


def steps_between(rate: float, prediction_rate: float) -> int:
    """The number of filter steps between two observations at `rate`."""

    return max(1, round(prediction_rate / rate))


def observe(
    positions: np.ndarray, rate: float, prediction_rate: float, noise: float, random: np.random.Generator
) -> np.ndarray:
    """The noisy positions a tracker observes at `rate`, NaN at the filter steps in between."""

    observations = np.full_like(positions, np.nan)
    observed = np.arange(0, len(positions), steps_between(rate, prediction_rate))
    observations[observed] = positions[observed] + random.normal(0, noise, (len(observed), 2))
    return observations


def animate(data: Path, promises: dict, output: Path, args: argparse.Namespace) -> None:
    """Renders CoFi tracking one flight per speed side by side as a GIF."""

    image = Image.open(data / "scene.png").convert("L")
    image_size = load_scene(data)[2]
    image_extent = (-image_size[0] / 2, image_size[0] / 2, -image_size[1] / 2, image_size[1] / 2)
    half_width, half_height = AREA_SIZE[0] / 2, AREA_SIZE[1] / 2
    xs, ys = np.meshgrid(np.linspace(-half_width, half_width, 500), np.linspace(-half_height, half_height, 440))
    grid = np.column_stack([xs.ravel(), ys.ravel()])

    figure, axes = plt.subplots(1, len(SPEEDS), figsize=(15, 4.7), sharey=True)
    figure.patch.set_facecolor("white")
    panels = []
    for axis, speed in zip(axes, SPEEDS):
        timestamps, truth, accelerations = interpolate(*load_flight(typical_flight(data, speed)), args.prediction_rate)
        random = np.random.default_rng(args.seed)
        observations = observe(truth, args.rate, args.prediction_rate, args.noise, random)

        # The noise of each IMU sample follows from its noise density and rate
        imu_noise = args.imu_density * np.sqrt(args.prediction_rate)
        accelerations = accelerations + random.normal(0, imu_noise, accelerations.shape)

        # The filter steps with the IMU and corrects its estimate whenever the tracker observes the drone
        cofi = ConstitutionalFilter(
            promises[speed],
            interpolation_method="hybrid",
            process_noise=imu_noise,
            # CoFi scales the sensor noise with the time since the last observation
            sensor_std=args.noise * args.prediction_rate / steps_between(args.rate, args.prediction_rate),
            **PARAMETERS,
        )
        result = cofi.run(observations, timestamps, seed=args.seed, accelerations=accelerations)

        compliance = cofi.constitution(grid).reshape(xs.shape)
        axis.set_facecolor("dimgray")
        axis.imshow(image, cmap="gray", extent=image_extent, alpha=0.55, zorder=1)
        axis.imshow(
            compliance, origin="lower", extent=(-half_width, half_width, -half_height, half_height),
            cmap="coolwarm_r", vmin=0, vmax=1, alpha=0.45, zorder=2,
        )
        axis.set_xlim(-half_width, half_width)
        axis.set_ylim(-half_height, half_height)
        axis.set_aspect("equal")
        axis.set_xlabel("x / m", fontsize=13)
        axis.set_title(f"v = {speed} m/s", fontsize=13)
        axis.plot(*truth.T, color="white", lw=1, alpha=0.35, zorder=3)

        panels.append(
            dict(
                shown=np.random.default_rng(args.seed).choice(PARAMETERS["number_of_particles"], 1_500, replace=False),
                truth=truth,
                result=result,
                particles=axis.scatter([], [], s=5, lw=0, zorder=4),
                trail=axis.plot([], [], color=COLOR, lw=2, zorder=5)[0],
                estimate=axis.plot([], [], "o", color=COLOR, ms=9, mec="white", mew=1.2, zorder=7)[0],
                drone=axis.plot([], [], "o", color="white", ms=5, zorder=6)[0],
            )
        )
    axes[0].set_ylabel("y / m", fontsize=13)
    figure.subplots_adjust(left=0.05, right=0.99, bottom=0.12, top=0.93, wspace=0.06)

    frames = round(args.duration * args.fps)

    def update(frame: int) -> None:
        # All flights progress from start to end over the same frames, as in CoCo's video
        progress = frame / (frames - 1)
        for panel in panels:
            row = round(progress * (len(panel["truth"]) - 1))
            result, shown = panel["result"], panel["shown"]

            # A subset of the particles is drawn, each shaded by its weight
            colors = np.tile(to_rgba(COLOR), (len(shown), 1))
            colors[:, 3] = np.clip(0.45 * len(result.weights[row]) * result.weights[row, shown], 0, 0.6)
            panel["particles"].set_offsets(result.particles[row, shown, :2])
            panel["particles"].set_facecolors(colors)
            panel["trail"].set_data(*result.estimates[max(0, row - round(1.5 * args.prediction_rate)) : row + 1, :2].T)
            panel["estimate"].set_data([result.estimates[row, 0]], [result.estimates[row, 1]])
            panel["drone"].set_data([panel["truth"][row, 0]], [panel["truth"][row, 1]])

    with tempfile.TemporaryDirectory() as folder:
        video = Path(folder) / "drones.mp4"
        FuncAnimation(figure, update, frames=frames).save(video, writer=FFMpegWriter(fps=args.fps), dpi=70)
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                matplotlib.rcParams["animation.ffmpeg_path"], "-v", "error", "-y", "-i", str(video), "-vf",
                "split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                str(output),
            ],
            check=True,
        )
    plt.close(figure)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="The GIF to render")
    parser.add_argument("--data", type=Path, default=root / "data", help="The folder of flights and obstacles")
    parser.add_argument("--rate", type=float, default=3.0, help="Observations per second, rounded to whole log rows")
    parser.add_argument("--noise", type=float, default=0.15, help="Observation noise per axis in meters")
    parser.add_argument("--prediction-rate", type=float, default=50.0, help="Filter steps and IMU measurements per second")
    parser.add_argument("--imu-density", type=float, default=0.16, help="IMU noise density per axis in m/s²/√Hz")
    parser.add_argument("--seed", type=int, default=0, help="Seeds the observation noise and the filter")
    parser.add_argument("--duration", type=float, default=12.0, help="Length of the animation in seconds")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second of the animation")
    args = parser.parse_args()

    scene, _, _ = load_scene(args.data)
    promises = {speed: make_promis(scene, speed) for speed in SPEEDS}
    animate(args.data, promises, args.output, args)


if __name__ == "__main__":
    main()
