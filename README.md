# Lennard-Jones-potential
This repository contains python code for a Lennard-Jones (LJ) potential. The LJ potential is a MD models short-range repulsion and intermediate-range attraction. A Metropolis Monte Carlo algorithm generates configurations according to their relative concentrations in a Boltzmann distribution. The current implementation provides a baseline for exploring how temperature and density influnece particle energy and configuration. 

The current model abides by U(r) = 4epsilon*((sigma/r)^12 - (sigma/r)^6). By setting epsilon = sigma = 1, we get a reduced potential of U(r) = 4(r^-12 - r^-6), where r is the seperation between two particles. The simulation uses a square box with periodic boundary conditions and uses the minimum-image conventiont using the nearest periodic image.

Each Monte Carlo step is accepted with the following rules: if ∆U <= 0 the move is accepted. Otherwise, the move accepted with a probability p = exp(-∆U/T). 

To install and use the repository:


Clone the repository and navigate to the project directory:
git clone https://github.com/lalitdas21/Lennard-Jones-potential.git
cd Lennard-Jones-potential

Create and activate a virtual enviornment:
python3 -m venv .venv
source .venv/bin/activate

On windows:
.venv\Scripts\activate

Install the dependencies:
pip install -r requirements.txt

To use this code, run:
python3 plot_results.py

The produced figure shows an example simulation with 36 particles at a reduced temperature of 1.0 in a square box of side length 6.0, sampled for 20,000 Monte Carlo steps. We can see that the potential energy initially fluctures for the first 4000-5000 steps until it settles down. This is consistent with the system exploring configurations with a more favorable interactions. Energy still fluctuates because the algorithm allows for exploration with a probability of exp(-∆U/T). This possibly suggests equilibrium, but not with certainty. The final configuration shows a roughly ordered arrangement of particles. This suggests that the Lennard-Jones potential favors certain particle seperatations. However, this does not establish whether or not the system is in an exact gas-like, liquid-like, or solid-like state. Additional simulations are necessary to determine this component of the behavior. 

To run the testing suite, run:
pytest -v

The tests include tests for minimum-image convention, Lennard-Jones potential values, local and total energy consistency, and simulation output properties.

Limitations and future work:

Current limitations include: untruncated Lennard-Jones potential, small two-dimensional systems that may show finite-size effects, no dedicated phase-identification procude. 

Future work may be done in implementing the radial distribution function, (g(r)), to quantify spatial correlations between particles, investigating equilibration, sampling uncertainty, and finite-size effects.