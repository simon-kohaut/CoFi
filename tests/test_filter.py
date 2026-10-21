#
# Copyright (c) Simon Kohaut, Felix Divo, and contributors
#
# This file is part of CoFi and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with CoFi.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

from time import sleep
from unittest import TestCase, main

import numpy as np
from scipy.stats import norm, uniform

from cofi import ConstitutionalFilter

N = 2_000
WIDTH, HEIGHT = 19_490.0, 19_988.0


def reference_systematic_resample(weights):
    """`filterpy.monte_carlo.systematic_resample`, as used by the original notebook."""

    n = len(weights)
    positions = (np.random.random() + np.arange(n)) / n
    indexes = np.zeros(n, "i")
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def reference_run(data, seed_value, constitutional_trust, interpolator):
    """The particle filter of the original CoFi notebook (`run_experiment`), without plotting.

    `data` has shape (T, 5) as (east, north, course, speed, dt), as built by the notebook.
    """

    def create_initial_particles(n):
        return np.stack(
            [
                uniform(loc=-WIDTH / 2, scale=WIDTH).rvs(n),
                uniform(loc=-HEIGHT / 2, scale=HEIGHT).rvs(n),
                norm(loc=0, scale=8).rvs(n),
                norm(loc=0, scale=8).rvs(n),
            ]
        ).T

    def predict(particles, dt, process_noise):
        particles = particles.copy()
        n = len(particles)
        scaled_process_noise = process_noise * dt
        p = particles[:, [0, 1]]
        v = particles[:, [2, 3]]
        particles[:, [0, 1]] = p + v * dt
        particles[:, [2, 3]] = v + np.random.normal(
            (0, 0), (scaled_process_noise, scaled_process_noise), size=(n, 2)
        )
        return particles

    def update(particles, weights, z, R, dt):
        weights = weights.copy()
        positions = particles[:, :2]
        distance = np.linalg.norm(positions - z, axis=1)
        if dt > 0.01:
            weights *= norm.pdf(distance, 0, R * dt)
        if constitutional_trust > 0:
            constitution = interpolator(positions)[:, 0]
            weight_change = constitutional_trust * constitution + (1 - constitutional_trust)
            weight_change[~np.isfinite(constitution)] = 0
            weights *= weight_change
        weights += 1.0e-300
        weights /= sum(weights)
        return weights

    np.random.seed(seed_value)
    particles = create_initial_particles(N)
    particles[:, 0] = norm(loc=data[0, 0], scale=10 * 60).rvs(N)
    particles[:, 1] = norm(loc=data[0, 1], scale=10 * 60).rvs(N)
    weights = np.ones(N) / N

    estimates = []
    for obs in data:
        dt = obs[4]
        particles = predict(particles, dt=dt, process_noise=0.02)
        weights = update(particles, weights, z=obs[:2], R=5, dt=dt)
        if 1.0 / np.sum(np.square(weights)) < N / 2:
            indexes = reference_systematic_resample(weights)
            particles[:] = particles[indexes]
            weights.resize(len(particles))
            weights.fill(1.0 / len(weights))
        estimates.append(np.average(particles, weights=weights, axis=0))

    return np.array(estimates)


def synthetic_track(steps=60, seed=7):
    """A vessel heading north-east at about 5 m/s, observed roughly every minute."""

    random = np.random.default_rng(seed)
    timestamps = np.cumsum(random.uniform(40, 80, steps))
    duplicates = np.array([steps // 6, steps // 2])
    timestamps[duplicates] = timestamps[duplicates - 1]  # duplicate reports
    timestamps -= timestamps[0]
    positions = np.stack([-3_000 + 4.0 * timestamps, -2_000 + 3.0 * timestamps]).T
    positions += random.normal(0, 20, positions.shape)
    return positions, timestamps


def constitution(positions):
    """Compliance falling off to the south-east, undefined far east, shape (N, 1)."""

    compliance = 1 / (1 + np.exp((positions[:, 0] - positions[:, 1]) / 500))
    compliance[positions[:, 1] < -2_500] = np.nan
    return compliance[:, None]


class Landscape:
    """Stands in for a landscape of ProMis, interpolated by `interpolator` whatever the method."""

    def __init__(self, interpolator):
        self.interpolator = interpolator

    def get_interpolator(self, interpolation_method):
        return self.interpolator


class FakeProMis:
    """Stands in for ProMis, handing out a landscape per `update`, and None once there are no more."""

    def __init__(self, *interpolators):
        self.updates = [None if interpolator is None else Landscape(interpolator) for interpolator in interpolators]

    def update(self):
        return self.updates.pop(0) if self.updates else None


class TestConstitutionalFilter(TestCase):
    def test_matches_original_notebook(self):
        """For a plain particle filter and for CoFi, estimates match the original notebook."""

        positions, timestamps = synthetic_track()
        data = np.column_stack(
            [positions, np.zeros(len(positions)), np.zeros(len(positions)), np.diff(timestamps, prepend=0)]
        )

        for trust in (0.0, 0.5, 1.0):
            expected = reference_run(data, seed_value=2023, constitutional_trust=trust, interpolator=constitution)

            # The notebook draws (and discards) uniform initial positions first
            random = np.random.RandomState(2023)
            random.uniform(0.0, 1.0, N)
            random.uniform(0.0, 1.0, N)
            result = ConstitutionalFilter(FakeProMis(constitution), trust=trust).run(positions, timestamps, random=random)

            np.testing.assert_allclose(result.estimates, expected, rtol=1e-9, atol=1e-6, err_msg=f"trust {trust}")

    def test_trust_zero_ignores_constitution(self):
        positions, timestamps = synthetic_track()
        plain = ConstitutionalFilter(FakeProMis()).run(positions, timestamps, seed=1)
        untrusted = ConstitutionalFilter(FakeProMis(constitution), trust=0.0).run(positions, timestamps, seed=1)
        np.testing.assert_array_equal(plain.estimates, untrusted.estimates)

    def test_result(self):
        positions, timestamps = synthetic_track(steps=20)
        result = ConstitutionalFilter(FakeProMis(constitution), trust=0.5, number_of_particles=300).run(
            positions, timestamps, seed=3
        )

        self.assertEqual(result.particles.shape, (20, 300, 4))
        self.assertEqual(result.weights.shape, (20, 300))
        self.assertEqual(result.estimates.shape, (20, 4))
        np.testing.assert_allclose(result.weights.sum(axis=1), 1.0)
        self.assertEqual(result.position_error.shape, (20,))
        self.assertEqual(result.as_dict()["pos_error_mean"], result.position_error.mean())

        # Particles where the constitution is undefined carry no weight
        undefined = np.isnan(constitution(result.particles[0, :, :2])[:, 0])
        self.assertTrue(undefined.any())
        self.assertTrue((result.weights[0][undefined] < 1e-200).all())

    def test_missing_observations(self):
        """Without observations, particles are only predicted and weighted by the constitution."""

        positions, timestamps = synthetic_track(steps=30)
        positions[10:20] = np.nan
        cofi = ConstitutionalFilter(FakeProMis(constitution), trust=0.5, number_of_particles=300)
        result = cofi.run(positions, timestamps, seed=3)

        self.assertTrue(np.isfinite(result.estimates).all())
        self.assertTrue(np.isnan(result.position_error[10:20]).all())
        self.assertEqual(result.as_dict()["pos_error_mean"], np.nanmean(result.position_error))

        # A plain particle filter keeps its weights while observations are missing
        plain = ConstitutionalFilter(FakeProMis(), number_of_particles=300).run(positions, timestamps, seed=3)
        np.testing.assert_allclose(plain.weights[10:20], plain.weights[[9] * 10])

    def test_accelerations(self):
        """Measured accelerations move the particles' velocities; zero ones change nothing."""

        positions, timestamps = synthetic_track(steps=20)
        cofi = ConstitutionalFilter(FakeProMis(constitution), trust=0.5, number_of_particles=300)
        plain = cofi.run(positions, timestamps, seed=3)
        still = cofi.run(positions, timestamps, seed=3, accelerations=np.zeros_like(positions))
        np.testing.assert_array_equal(plain.estimates, still.estimates)

        # Without observations, the estimate follows the measured acceleration
        positions = np.full((11, 2), np.nan)
        positions[0] = 0.0
        timestamps = np.arange(11.0)
        accelerations = np.tile([1.0, 0.0], (11, 1))
        result = ConstitutionalFilter(FakeProMis(), number_of_particles=2_000, process_noise=0.0,
                                      initial_position_std=0.0, initial_velocity_std=0.0).run(
            positions, timestamps, seed=3, accelerations=accelerations
        )
        np.testing.assert_allclose(result.estimates[-1], [45.0, 0.0, 10.0, 0.0], atol=1e-9)

    def test_constitution_interval(self):
        """With a constitution interval, the constitution's influence does not depend on the step rate."""

        def run(steps_per_second):
            timestamps = np.arange(0, 1 + 1e-9, 1 / steps_per_second)
            positions = np.full((len(timestamps), 2), np.nan)
            positions[0] = 0.0
            cofi = ConstitutionalFilter(
                FakeProMis(lambda xy: (xy[:, 0] > 0).astype(float) * 0.8 + 0.1), trust=1.0, number_of_particles=1_000,
                process_noise=0.0, initial_position_std=1.0, initial_velocity_std=0.0, constitution_interval=1.0,
            )
            result = cofi.run(positions, timestamps, random=np.random.RandomState(0))
            return (result.weights[-1] * (result.particles[-1, :, 0] > 0)).sum()

        # One second of weighting by 0.9 against 0.1 leaves 90 % of the weight right of zero,
        # whether in 10 or 100 steps per second
        np.testing.assert_allclose(run(10), 0.9, atol=0.02)
        np.testing.assert_allclose(run(100), 0.9, atol=0.02)

    def test_online_matches_run(self):
        """Stepping online gives the same track as running offline."""

        positions, timestamps = synthetic_track(steps=20)
        positions[5:8] = np.nan
        accelerations = np.random.default_rng(0).normal(0, 0.1, positions.shape)
        cofi = ConstitutionalFilter(FakeProMis(constitution), trust=0.5, number_of_particles=300)
        result = cofi.run(positions, timestamps, seed=3, accelerations=accelerations)

        cofi.start(positions[0], timestamps[0], seed=3)
        estimates = [cofi.step(*row) for row in zip(positions, timestamps, accelerations)]
        np.testing.assert_array_equal(estimates, result.estimates)
        np.testing.assert_array_equal(cofi.particles, result.particles[-1])
        np.testing.assert_array_equal(cofi.weights, result.weights[-1])

    def test_online_promis_updates(self):
        """The filter follows the landscapes ProMis delivers, and keeps the last one in between."""

        promis = FakeProMis(lambda xy: np.ones(len(xy)), lambda xy: (xy[:, 0] > 0).astype(float))
        cofi = ConstitutionalFilter(
            promis, trust=1.0, number_of_particles=1_000, process_noise=0.0,
            initial_position_std=1.0, initial_velocity_std=0.0,
        )
        cofi.start([0.0, 0.0], 0.0, seed=0)
        cofi.step([np.nan, np.nan], 0.0)
        self.assertLess(abs(cofi.estimate[0]), 0.2)

        # The updated landscape rules out everything left of zero
        cofi.step([np.nan, np.nan], 1.0)
        self.assertTrue((cofi.particles[cofi.weights > 1e-200, 0] > 0).all())

        landscape = cofi.constitution
        cofi.step([np.nan, np.nan], 2.0)
        self.assertIs(cofi.constitution, landscape)

    def test_promis(self):
        """With a landscape from ProMis, the particles keep off the obstacle the constitution forbids."""

        from promis import ProMis, StaRMap
        from promis.geo import CartesianMap, CartesianPolygon, CartesianRasterBand, PolarLocation

        logic = """
        over(obstacle) <- source("/star_map/over/obstacle", Probability).
        landscape if not over(obstacle).
        landscape -> target("/landscape").
        """
        obstacle = CartesianPolygon.from_numpy(
            np.array([[-1, 1, 1, -1], [-0.5, -0.5, 0.5, 0.5]]), location_type="obstacle"
        )
        area = CartesianMap(PolarLocation(0, 0), {"obstacle": [obstacle]})
        area.apply_covariance({"obstacle": 0.01**2 * np.eye(2)})
        points = CartesianRasterBand(PolarLocation(0, 0), (40, 40), 8, 8)
        np.random.seed(0)
        star_map = StaRMap(area)
        star_map.initialize(points, number_of_random_maps=10, logic=logic)
        promis = ProMis(star_map, logic, 40 * 40)
        promis.initialize(points)
        sleep(0.05)  # Resin's writers deliver values asynchronously

        cofi = ConstitutionalFilter(
            promis, trust=1.0, number_of_particles=1_000, initial_position_std=1.0, initial_velocity_std=0.0
        )
        cofi.start([0.0, 0.0], 0.0, seed=0)
        cofi.step([np.nan, np.nan], 0.0)
        self.assertIsNotNone(cofi.constitution)

        on_obstacle = (np.abs(cofi.particles[:, 0]) < 0.9) & (np.abs(cofi.particles[:, 1]) < 0.4)
        self.assertGreater(on_obstacle.mean(), 0.1)
        self.assertLess(cofi.weights[on_obstacle].sum(), 0.01)

    def test_online_order(self):
        cofi = ConstitutionalFilter(FakeProMis(), number_of_particles=10)
        with self.assertRaises(RuntimeError):
            cofi.step([0.0, 0.0], 0.0)
        with self.assertRaises(RuntimeError):
            cofi.estimate

        cofi.start([0.0, 0.0], 1.0, seed=0)
        with self.assertRaises(ValueError):
            cofi.step([0.0, 0.0], 0.5)


if __name__ == "__main__":
    main()
