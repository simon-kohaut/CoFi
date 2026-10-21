"""The Constitutional Filter, a particle filter that trusts agents to follow their constitution."""

#
# Copyright (c) Simon Kohaut, Felix Divo, and contributors
#
# This file is part of CoFi and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with CoFi.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

# Third Party
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

if TYPE_CHECKING:
    from promis import ProMis


@dataclass
class FilterResult:
    """The trace of a filter run over `T` observations with `N` particles.

    Attributes:
        particles: The particles after each step, shape (T, N, 4) as (east, north, v_east, v_north).
        weights: The particle weights after each step, shape (T, N).
        estimates: The weighted mean state after each step, shape (T, 4).
        truth: The observed positions, shape (T, 2).
    """

    particles: NDArray
    weights: NDArray
    estimates: NDArray
    truth: NDArray

    @property
    def position_error(self) -> NDArray:
        """The distance between estimated and observed position at each step, shape (T,).

        Steps without an observation are NaN.
        """

        return np.linalg.norm(self.estimates[:, :2] - self.truth, axis=1)

    def as_dict(self) -> dict:
        """The result as a flat dict, e.g., for collecting many runs in a DataFrame."""

        position_error = self.position_error
        return {
            "particles": self.particles,
            "weights": self.weights,
            "estimates": self.estimates,
            "truth": self.truth,
            "pos_error": position_error,
            "pos_error_mean": np.nanmean(position_error),
        }


class ConstitutionalFilter:
    """A particle filter whose belief is weighted by an agent's constitution.

    The state of each particle is its position and velocity in a local Cartesian frame,
    propagated with a constant-velocity model. Besides the likelihood of the observed
    position, each particle's weight is scaled by `trust * c + (1 - trust)`, where `c` is the
    probability that the agent complies with its constitution at the particle's position,
    interpolated from the latest landscape of ProMis. With a trust of 0, or until ProMis has
    delivered a landscape, this is a plain particle filter; positions where the landscape is
    undefined (non-finite) are ruled out.

    At the beginning of each step, the filter checks ProMis for an updated landscape with
    `promis.update()`, which hands out each landscape only once; leave these calls to the filter.
    The landscape must cover the particles in the frame and units of the observed positions.
    The filter tracks an agent either offline with `run`, or online with `start` and `step`.

    Args:
        promis: The ProMis instance that computes the landscapes of the agent's constitution.
        trust: How much the filter trusts the agent to follow its constitution, in [0, 1].
        number_of_particles: The number of particles.
        sensor_std: The standard deviation of a position observation in meters per second
            since the last observation.
        process_noise: The standard deviation of velocity changes in meters per second squared;
            with measured accelerations, e.g., from an IMU, the noise of these measurements.
        initial_position_std: The spread of initial positions around the first observation in meters.
        initial_velocity_std: The spread of initial velocities around zero in meters per second.
        constitution_interval: The seconds over which the constitution weights the particles once,
            so that its influence does not depend on how often the filter steps; by default, it
            weights them once per step.
        interpolation_method: How the landscapes are interpolated, e.g., "linear", which leaves
            them undefined outside of their evaluation points, or "hybrid", which falls back to
            the nearest evaluation point there.
    """

    def __init__(
        self,
        promis: "ProMis",
        trust: float = 0.5,
        number_of_particles: int = 2_000,
        sensor_std: float = 5.0,
        process_noise: float = 0.02,
        initial_position_std: float = 600.0,
        initial_velocity_std: float = 8.0,
        constitution_interval: float | None = None,
        interpolation_method: str = "linear",
    ):
        self.promis = promis
        self.trust = trust
        self.number_of_particles = number_of_particles
        self.sensor_std = sensor_std
        self.process_noise = process_noise
        self.initial_position_std = initial_position_std
        self.initial_velocity_std = initial_velocity_std
        self.constitution_interval = constitution_interval
        self.interpolation_method = interpolation_method

        # The latest landscape of ProMis, interpolated
        self.constitution: Callable[[NDArray], NDArray] | None = None

        # The state of an online track, set by `start`
        self.particles: NDArray | None = None
        self.weights: NDArray | None = None
        self.random: np.random.RandomState | None = None
        self.timestamp: float | None = None
        self.last_observation: float | None = None
        self.acceleration = np.zeros(2)

    @property
    def estimate(self) -> NDArray:
        """The weighted mean state of the particles, shape (4,) as (east, north, v_east, v_north)."""

        if self.particles is None:
            raise RuntimeError("The filter has not been started; call `start` first")

        return np.average(self.particles, weights=self.weights, axis=0)

    def start(
        self,
        position: NDArray,
        timestamp: float,
        seed: int | None = None,
        random: np.random.RandomState | None = None,
    ) -> None:
        """Starts tracking an agent online, spreading the particles around its first observed position.

        The first observation itself is then passed to `step` like any other.

        Args:
            position: The first observed position, shape (2,) as (east, north) in meters.
            timestamp: The time of the first observation in seconds.
            seed: Seeds the random number generator if `random` is not given.
            random: The random number generator to use.
        """

        self.random = random if random is not None else np.random.RandomState(seed)
        self.particles = self.initial_particles(np.asarray(position, dtype=float), self.random)
        self.weights = np.full(self.number_of_particles, 1.0 / self.number_of_particles)
        self.timestamp = float(timestamp)
        self.last_observation = float(timestamp)
        self.acceleration = np.zeros(2)

    def step(self, position: NDArray, timestamp: float, acceleration: NDArray | None = None) -> NDArray:
        """Moves the particles to `timestamp` and weights them by the observation and the constitution.

        Args:
            position: The observed position, shape (2,) as (east, north) in meters; non-finite
                if missing, where only the constitution weights the predicted particles.
            timestamp: The time of this step in seconds, no earlier than the previous one.
            acceleration: The acceleration of the agent measured at `timestamp`, shape (2,) in meters
                per second squared, e.g., from an IMU, which moves the particles' velocities until the
                next step; `None` keeps their velocities constant up to the process noise.

        Returns:
            The estimated state after this step, shape (4,) as (east, north, v_east, v_north).
        """

        if self.particles is None:
            raise RuntimeError("The filter has not been started; call `start` first")
        if timestamp < self.timestamp:
            raise ValueError(f"Timestamp {timestamp} is earlier than the previous step at {self.timestamp}")

        # Follow the latest landscape, if ProMis has computed a new one
        landscape = self.promis.update()
        if landscape is not None:
            self.constitution = landscape.get_interpolator(self.interpolation_method)

        position = np.asarray(position, dtype=float)
        dt = timestamp - self.timestamp

        # The acceleration measured at the previous step acts until this one
        self.particles = self.predict(self.particles, dt, self.random, self.acceleration)
        self.weights = self.update(self.particles, self.weights, position, timestamp - self.last_observation, dt)
        self.timestamp = timestamp
        self.acceleration = np.zeros(2) if acceleration is None else np.asarray(acceleration, dtype=float)
        if np.all(np.isfinite(position)):
            self.last_observation = timestamp

        # Resample if too few particles carry the weight
        if 1.0 / np.sum(np.square(self.weights)) < self.number_of_particles / 2:
            self.particles = self.particles[systematic_resample(self.weights, self.random)]
            self.weights = np.full(self.number_of_particles, 1.0 / self.number_of_particles)

        return self.estimate

    def run(
        self,
        positions: NDArray,
        timestamps: NDArray,
        seed: int | None = None,
        random: np.random.RandomState | None = None,
        accelerations: NDArray | None = None,
    ) -> FilterResult:
        """Tracks an agent through a sequence of observed positions, starting a new track.

        Args:
            positions: The observed positions, shape (T, 2) as (east, north) in meters;
                non-finite rows mark missing observations, where only the constitution
                weights the predicted particles.
            timestamps: The time of each observation in seconds, shape (T,).
            seed: Seeds the random number generator if `random` is not given.
            random: The random number generator to use.
            accelerations: Measured accelerations of the agent, shape (T, 2) in meters per second
                squared, e.g., from an IMU, which move the particles' velocities between the
                timestamps; `None` keeps their velocities constant up to the process noise.

        Returns:
            The trace of particles, weights and estimates over all observations.
        """

        positions = np.asarray(positions, dtype=float)
        timestamps = np.asarray(timestamps, dtype=float)
        if accelerations is None:
            accelerations = np.zeros_like(positions)

        self.start(positions[0], timestamps[0], seed=seed, random=random)

        particle_trace, weight_trace, estimates = [], [], []
        for position, timestamp, acceleration in zip(positions, timestamps, np.asarray(accelerations, dtype=float)):
            estimates.append(self.step(position, timestamp, acceleration))
            particle_trace.append(self.particles.copy())
            weight_trace.append(self.weights.copy())

        return FilterResult(
            particles=np.array(particle_trace),
            weights=np.array(weight_trace),
            estimates=np.array(estimates),
            truth=positions,
        )

    def initial_particles(self, first_position: NDArray, random: np.random.RandomState) -> NDArray:
        """Particles with zero-mean velocities, spread around the first observed position."""

        n = self.number_of_particles
        velocity_east = norm(loc=0, scale=self.initial_velocity_std).rvs(n, random_state=random)
        velocity_north = norm(loc=0, scale=self.initial_velocity_std).rvs(n, random_state=random)
        east = norm(loc=first_position[0], scale=self.initial_position_std).rvs(n, random_state=random)
        north = norm(loc=first_position[1], scale=self.initial_position_std).rvs(n, random_state=random)

        return np.stack([east, north, velocity_east, velocity_north]).T

    def predict(
        self,
        particles: NDArray,
        dt: float,
        random: np.random.RandomState,
        acceleration: NDArray | None = None,
    ) -> NDArray:
        """Moves the particles for `dt` seconds, accelerating them by a noisy `acceleration`."""

        particles = particles.copy()
        noise = self.process_noise * dt
        particles[:, [0, 1]] = particles[:, [0, 1]] + particles[:, [2, 3]] * dt
        if acceleration is not None:
            particles[:, [2, 3]] = particles[:, [2, 3]] + np.asarray(acceleration) * dt
        particles[:, [2, 3]] = particles[:, [2, 3]] + random.normal(
            (0, 0), (noise, noise), size=(len(particles), 2)
        )

        return particles

    def update(
        self, particles: NDArray, weights: NDArray, position: NDArray, dt: float, step: float | None = None
    ) -> NDArray:
        """Weights the particles by the observed position and the agent's constitution.

        Args:
            particles: The predicted particles, shape (N, 4).
            weights: The current particle weights, shape (N,).
            position: The observed position, shape (2,); non-finite if missing.
            dt: The seconds since the previous observation, which scale the sensor noise.
            step: The seconds since the previous step, which scale the constitution's influence
                if a `constitution_interval` is set.
        """

        weights = weights.copy()
        locations = particles[:, :2]

        # Observations arriving at (almost) the same time carry no new information,
        # and missing observations none at all
        if dt > 0.01 and np.all(np.isfinite(position)):
            distance = np.linalg.norm(locations - position, axis=1)
            weights *= norm.pdf(distance, 0, self.sensor_std * dt)

        if self.constitution is not None and self.trust > 0:
            compliance = np.asarray(self.constitution(locations), dtype=float).reshape(len(locations), -1)[:, 0]
            weight_change = self.trust * compliance + (1 - self.trust)
            weight_change[~np.isfinite(compliance)] = 0
            if self.constitution_interval is not None and step is not None:
                weight_change = weight_change ** (step / self.constitution_interval)
            weights *= weight_change

        weights += 1.0e-300  # avoid round-off to zero
        weights /= np.sum(weights)

        return weights


def systematic_resample(weights: NDArray, random: np.random.RandomState) -> NDArray:
    """Indices of particles drawn by systematic resampling according to `weights`."""

    n = len(weights)
    positions = (random.random() + np.arange(n)) / n
    indices = np.searchsorted(np.cumsum(weights), positions, side="right")

    return np.minimum(indices, n - 1)
