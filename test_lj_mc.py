import numpy as np
import pytest

from lj_mc import (
    lj_potential,
    minimum_image,
    particle_energy,
    total_energy,
    simulate,
)

R_MIN_SQUARED = 2.0 ** (1.0 / 3.0)  # r^2 = 2^(1/3), the LJ minimum

class TestMinimumImage:
    def test_small_displacement(self):
        """ In a periodic boundary condition system, the minimum-image convension
            places particles in a [-L/2, L/2] range. Here, that range in [-2.5, 2.5]
            and thus the particles at points 0.4, 0.5 should not be moved  """
        d = np.array([0.4, 0.5])
        assert np.allclose(minimum_image(d, 5.0), d)

    def test_wraps_to_nearest_image(self):
        """ In this test, we have a box of length 6 with particles are 5.4 and -5.4. These particles 
            by convention must be wrapped to 5.4 - 6.0 * 6(5.4/6.0) = -0.6. Likewise, we get 0.6 
            for -5.4. """
        assert minimum_image(np.array([5.4]), 6.0)[0] == pytest.approx(-0.6)
        assert minimum_image(np.array([-5.4]), 6.0)[0] == pytest.approx(0.6)

    def test_result_within_half_box(self):
        """ In this test, we run a property-based test for 1000 arrays, each with 1000 2d positions
            for a total of 2 million coordinates to check that minimum_image always returns values in 
            the specified range of [-L/2, L/2]. """
        rng = np.random.default_rng(0)
        for _ in range(1000):
            d = rng.uniform(-50, 50, size=(1000, 2))
            wrapped = minimum_image(d, 6.0)
            assert np.all(np.abs(wrapped) <= 3.0)

    def test_differs_from_input_by_whole_boxes(self): 
        """ In this test, we run a property-based test for 1,000 arrays, each containing
            1,000 2D displacement vectors, for a total of 2 million coordinates. We check
            that the difference between each original displacement and its minimum-image
            displacement is an integer multiple of the box length, as required by periodic
            boundary conditions. """
        rng = np.random.default_rng(1) 
        for _ in range(1000):
            d = rng.uniform(-50, 50, size=(1000, 2)) 
            shifts = (d - minimum_image(d, 6.0)) / 6.0 
            assert np.allclose(shifts, np.rint(shifts))

class TestLJPotential:
    def test_zero_crossing_at_sigma(self):
        """ At sigma = 1, there should be a zero crossing, and thus the potential function
            must be zero there. """
        assert lj_potential(1.0) == pytest.approx(0.0, abs=1e-12)

    def test_minimum_depth_is_minus_epsilon(self):
        """ At r = 2^(1/6), which is r^2 = 2^(1/3), the Lennard_Jones potential should be minimal
            and equal to -1. This tests that it is approximately -1.0 """
        assert lj_potential(R_MIN_SQUARED) == pytest.approx(-1.0)

    def test_minimum_depth_is_a_minimum(self):
        """ At r = 2^(1/6), which is r^2 = 2^(1/3), the Lennard_Jones potential should be minimal
            and equal to -1. This tests that it at least a local minimum. """
        nearby = lj_potential(np.array([0.98, 1.02]) * R_MIN_SQUARED)
        assert np.all(nearby > lj_potential(R_MIN_SQUARED))

    def test_known_value(self):
        """ At r^2 = 4, we know that the potential should be 4 * (1/2^12 -1/2^6). """
        assert lj_potential(4.0) == pytest.approx(4 * (2**-12 - 2**-6))

    def test_decays_to_zero_at_large_separation(self):
        """ As r^2 goes to infinity, we know that the potential should decay to 0. """
        assert abs(lj_potential(100.0**2)) < 1e-10

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_nonpositive_scalar(self, bad):
        """ Test that lj_potential raises ValueError when given a zero or negative squared separation, 
            which is physically invalid for the Lennard-Jones potential. """
        with pytest.raises(ValueError):
            lj_potential(bad)
 
    def test_rejects_array_containing_a_zero(self):
        """ Test that lj_potential raises ValueError when an array of squared separations contains a zero, 
            preventing division by zero in the potential calculation. """
        with pytest.raises(ValueError):
            lj_potential(np.array([1.0, 0.0, 2.0]))

def lattice_with_jitter(rng, side=3, box=6.0, jitter=0.2):
    """ Generates a slightly perturbed square lattice of particles. Small displacements
        disrupt perfect symmetry while preventing overlaps. """
    spacing = box / side
    grid = np.arange(side) * spacing + spacing / 2
    pos = np.array([(x, y) for x in grid for y in grid])
    return pos + rng.uniform(-jitter, jitter, size=pos.shape)

class TestTotalEnergy:
    def test_single_particle_has_zero_energy(self):
        """ If a particle has no neighbors, particle_energy should be 0 """
        assert total_energy(np.array([[1.0, 1.0]]), 6.0) == 0.0

    def test_two_particles_match_pair_formula(self):
        """ If a particle has one neighbor, its energy should be the Lennard-Jones potential"""
        pos = np.array([[1.0, 1.0], [2.5, 1.0]])  
        assert total_energy(pos, 20.0) == pytest.approx(lj_potential(1.5**2))
 
    def test_pair_counted_once_not_twice(self):
        """ Verifies that each particle pair only contributes once to the energy. """
        pos = np.array([[1.0, 1.0], [2.5, 1.0]])
        assert total_energy(pos, 20.0) != pytest.approx(2 * lj_potential(1.5**2))
 
    def test_periodic_boundary_is_used(self):
        """ Verifies that particles use the periodic boundaries as reflected by minimum image. """
        pos = np.array([[0.6, 3.0], [5.4, 3.0]])
        assert total_energy(pos, 6.0) == pytest.approx(lj_potential(1.2**2))
 
    def test_three_particles_sum_all_three_pairs(self):
        """ Verifies that the energies for three particles is reflected by the sum of those three particles. """
        pos = np.array([[1.0, 1.0], [2.2, 1.0], [1.0, 2.5]])
        d = lambda a, b: np.sum((pos[a] - pos[b]) ** 2)
        expected = (
            lj_potential(d(0, 1)) + lj_potential(d(0, 2)) + lj_potential(d(1, 2))
        )
        assert total_energy(pos, 50.0) == pytest.approx(expected)
 
    def test_invariant_under_particle_relabelling(self):
        """ Verifies that permuting particle order does not change the total energy. """
        rng = np.random.default_rng(3)
        pos = lattice_with_jitter(rng)
        shuffled = pos[rng.permutation(len(pos))]
        assert total_energy(pos, 6.0) == pytest.approx(total_energy(shuffled, 6.0))
 
    def test_invariant_under_global_translation(self):
        """ Veries that total energy does not change given a change in axes. """
        rng = np.random.default_rng(4)
        pos = lattice_with_jitter(rng)
        shifted = (pos + np.array([1.234, -0.777])) % 6.0
        assert total_energy(pos, 6.0) == pytest.approx(total_energy(shifted, 6.0))
 
    def test_overlapping_particles_raise(self):
        """ Veries that total energy raises and exception when there is overlap, as such an arrangement is not
            valid physically for the Lennard-Jones Potential. """
        pos = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(ValueError):
            total_energy(pos, 6.0)

class TestParticleEnergy:
    def test_no_neighbours_gives_zero(self):
        """ If a particle has no neighbors, particle_energy should be 0 """
        assert particle_energy(np.array([1.0, 1.0]), np.empty((0, 2)), 6.0) == 0.0

    def test_single_neighbour_matches_pair_formula(self):
        """ Tests that a particle has one neighbor, particle_energy should be equal to
            Lennard-Jones potential of their distance """
        e = particle_energy(np.array([1.0, 1.0]), np.array([[2.5, 1.0]]), 20.0)
        assert e == pytest.approx(lj_potential(1.5**2))

    def test_summing_over_particles_double_counts_total_energy(self):
        """ Tests that summing a particle's energy is double that of the total energy, since
            each pair, not particle, conributes to the total energy. It does this test over
            1000 random configurations. """
        rng = np.random.default_rng(0)
        for _ in range(1000): 
            pos = lattice_with_jitter(rng)
            per_particle = sum(
                particle_energy(pos[i], np.delete(pos, i, axis=0), 6.0)
                for i in range(len(pos))
            )
            assert per_particle == pytest.approx(2.0 * total_energy(pos, 6.0))

    def test_local_delta_equals_global_delta(self):
        """ Test that moving one particle changes the total system energy by the same amount as the change in that 
            particle's interaction energy with all other particles, as required by the Metropolis algorithm. """
        rng = np.random.default_rng(0)
        for _ in range(1000):
            pos = lattice_with_jitter(rng)
            i = 4
            others = np.delete(pos, i, axis=0)
            old, new = pos[i].copy(), (pos[i] + [-0.2, 0.2]) % 6.0

            local_delta = particle_energy(new, others, 6.0) - particle_energy(
                old, others, 6.0
            )

            moved = pos.copy()
            moved[i] = new
            global_delta = total_energy(moved, 6.0) - total_energy(pos, 6.0)

            assert local_delta == pytest.approx(global_delta)

    def test_overlap_raises(self):
        """ If two particles overlap, then it is not possible to compute the energy. In this case, an exception
            should be raised since having 0 seperation makes the Lennard-Jones Potential invalid. """
        with pytest.raises(ValueError):
            particle_energy(np.array([1.0, 1.0]), np.array([[1.0, 1.0]]), 6.0)

class TestSimulation:

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"particle_count": 0},
            {"temperature": 0.0},
            {"temperature": -1.0},
            {"box_length": 0.0},
            {"max_displacement": 0.0},
            {"steps": 0},
        ],
    )
    def test_bad_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            simulate(**kwargs)

    SMALL = dict(particle_count=16, box_length=5.0, steps=2000)

    def test_output_shapes(self):
        """ Test that the simulation returns positions with shape (N, 2) and an energy history containing one 
            initial value plus one value per Monte Carlo step. """
        r = simulate(**self.SMALL)
        assert r.positions.shape == (16, 2)
        assert r.energies.shape == (2001,)

    def test_non_square_particle_count(self):
        """ Test that the simulation correctly initializes and handles a particle count that is not a perfect square. """
        r = simulate(particle_count=10, box_length=5.0, steps=200)
        assert r.positions.shape == (10, 2)

    def test_same_seed_is_reproducible(self):
        """ Test that simulations with identical parameters and random seeds produce identical energy histories and 
            final particle positions. """
        a = simulate(seed=7, **self.SMALL)
        b = simulate(seed=7, **self.SMALL)
        assert np.array_equal(a.energies, b.energies)
        assert np.array_equal(a.positions, b.positions)

    def test_initial_energy_matches_lattice_energy(self):
        """ Test that the initial energy recorded by the simulation matches an independent energy calculation for 
            the initial square lattice. """
        r = simulate(**self.SMALL)
        side, box = 4, 5.0
        grid = np.arange(side) * (box / side) + (box / side) / 2
        lattice = np.array([(x, y) for x in grid for y in grid])
        assert r.energies[0] == pytest.approx(total_energy(lattice, box))

    def test_positions_stay_inside_box(self):
        """ Test that all final particle coordinates remain within the periodic simulation box after Monte Carlo moves. """
        r = simulate(**self.SMALL)
        assert np.all(r.positions >= 0.0)
        assert np.all(r.positions < 5.0)

    def test_acceptance_rate_is_a_probability(self):
        """Test that the reported fraction of accepted Monte Carlo moves lies between zero and one, inclusive. """
        r = simulate(**self.SMALL)
        assert 0.0 <= r.acceptance_rate <= 1.0

    def test_running_energy_matches_recomputed_energy(self):
        """ Test that the final tracked energy agrees with an independent recomputation from the final positions, 
            preventing errors in incremental energy updates. """
        r = simulate(**self.SMALL)
        assert r.energies[-1] == pytest.approx(
            total_energy(r.positions, 5.0), abs=1e-8
        )

    def test_energy_history_is_finite(self):
        """Verify that every recorded simulation energy is finite."""
        r = simulate(**self.SMALL)
        assert np.all(np.isfinite(r.energies))

    def test_lone_particle_always_accepts(self):
        """ Test the limiting case of a single particle, which has zero pair-interaction energy and accepts every proposed 
            move. """
        r = simulate(particle_count=1, box_length=5.0, steps=500)
        assert r.acceptance_rate == 1.0
        assert np.all(r.energies == 0.0)