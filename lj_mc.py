from dataclasses import dataclass
import numpy as np

def minimum_image(displacement: np.ndarray, box_length: float) -> np.ndarray:
    """Map displacement components to the nearest periodic image. """
    return displacement - box_length * np.rint(displacement / box_length)

def lj_potential(r_squared: float) -> float:
    """Lennard-Jones pair potential, given squared separation(s)."""
    r_squared = np.asarray(r_squared, dtype=float)
    if np.any(r_squared <= 0.0):
        raise ValueError("Particle separation must be positive.")

    inverse_r6 = (1.0 / r_squared) ** 3
    return 4.0 * (inverse_r6**2 - inverse_r6)

def total_energy(positions: np.ndarray, box_length: float) -> float:
    """Compute the total pair energy, counting each pair exactly once."""
    energy = 0.0
    particle_count = len(positions)

    for i in range(particle_count):
        displacement = minimum_image(
            positions[i + 1 :] - positions[i], box_length
        )
        distances_squared = np.sum(displacement**2, axis=1)

        if np.any(distances_squared == 0.0):
            raise ValueError("Two particles occupy the same position.")

        energy += np.sum(lj_potential(distances_squared))

    return float(energy)


def particle_energy(
    position: np.ndarray,
    other_positions: np.ndarray,
    box_length: float) -> float:
    """Energy of one particle interacting with all other particles."""
    
    if len(other_positions) == 0:
        return 0.0

    displacement = minimum_image(
        other_positions - position, box_length
    )
    distances_squared = np.sum(displacement**2, axis=1)

    if np.any(distances_squared == 0.0):
        raise ValueError("Two particles occupy the same position.")

    return float(np.sum(lj_potential(distances_squared)))

@dataclass
class SimulationResult:
    positions: np.ndarray
    energies: np.ndarray
    acceptance_rate: float


def simulate(
    particle_count: int = 36,
    temperature: float = 1.0,
    box_length: float = 6.0,
    max_displacement: float = 0.15,
    steps: int = 20_000,
    seed: int = 2026) -> SimulationResult:
    """Run a Metropolis Monte Carlo simulation."""
    if particle_count < 1:
        raise ValueError("particle_count must be positive.")
    if temperature <= 0.0 or box_length <= 0.0:
        raise ValueError("Temperature and box length must be positive.")
    if max_displacement <= 0.0 or steps < 1:
        raise ValueError("Displacement and step count must be positive.")

    rng = np.random.default_rng(seed)

    # Start from a regular lattice to avoid random initial overlaps.
    side = int(np.ceil(np.sqrt(particle_count)))
    spacing = box_length / side
    grid = np.arange(side) * spacing + spacing / 2.0
    positions = np.array(
        [(x, y) for x in grid for y in grid][:particle_count],
        dtype=float,
    )

    energy = total_energy(positions, box_length)
    energies = np.empty(steps + 1)
    energies[0] = energy
    accepted = 0

    for step in range(1, steps + 1):
        particle = rng.integers(particle_count)
        old_position = positions[particle].copy()

        # A uniform random displacement in each coordinate.
        displacement = rng.uniform(
            -max_displacement, max_displacement, size=2
        )
        new_position = (old_position + displacement) % box_length

        others = np.delete(positions, particle, axis=0)
        old_energy = particle_energy(
            old_position, others, box_length
        )
        new_energy = particle_energy(
            new_position, others, box_length
        )
        delta_energy = new_energy - old_energy

        # Accept downhill moves; accept uphill moves probabilistically.
        if (
            delta_energy <= 0.0
            or rng.random() < np.exp(-delta_energy / temperature)
        ):
            positions[particle] = new_position
            energy += delta_energy
            accepted += 1

        energies[step] = energy

    return SimulationResult(
        positions=positions.copy(),
        energies=energies,
        acceptance_rate=accepted / steps,
    )


if __name__ == "__main__":
    result = simulate()

    print(f"Final total energy: {result.energies[-1]:.6f}")
    print(f"Final energy per particle: "
          f"{result.energies[-1] / len(result.positions):.6f}")
    print(f"Acceptance rate: {result.acceptance_rate:.3f}")