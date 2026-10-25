import matplotlib.pyplot as plt
from lj_mc import simulate

result = simulate(steps=20_000, seed=2026)
particle_count = len(result.positions)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Energy history
axes[0].plot(result.energies / particle_count, linewidth=0.8)
axes[0].set_xlabel("Monte Carlo step")
axes[0].set_ylabel("Potential energy per particle")
axes[0].set_title("Energy during sampling")

# Final particle positions
axes[1].scatter(
    result.positions[:, 0],
    result.positions[:, 1],
    s=45,
)
axes[1].set_xlim(0, 6)
axes[1].set_ylim(0, 6)
axes[1].set_aspect("equal")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("Final particle configuration")

fig.tight_layout()
fig.savefig("results.png", dpi=180)
plt.show()

print(f"Acceptance rate: {result.acceptance_rate:.1%}")