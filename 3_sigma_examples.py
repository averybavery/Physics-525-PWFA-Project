import numpy as np
import matplotlib.pyplot as plt

# Physical Constants
e = 1.602176634e-19
epsilon_0 = 8.8541878128e-12
m_e = 9.1093837015e-31
c = 299792458.0

# Input Parameters
n_0 = 1.0e22      # plasma density [m^-3]
n_b0 = 2.0e21     # beam peak density [m^-3]

# Cases to compare
kp_sigma_values = [0.5, 1.0, 2.0]

xi_min = -2000e-6   
xi_max =  500e-6    
npts = 2**13

# Derived parameters
omega_p = np.sqrt(n_0 * e**2 / (m_e * epsilon_0))
k_p = omega_p / c
lambda_p = 2 * np.pi / k_p

print(f"lambda_p = {lambda_p*1e6:.2f} microns")
print(f"k_p = {k_p:.4e} 1/m")

# Gridding stuff
xi = np.linspace(xi_min, xi_max, npts)
dxi = xi[1] - xi[0]

# Correct Greens function
Xi = xi[:, None]
Xp = xi[None, :]

causal_mask = (Xp >= Xi).astype(float)
kernel = np.sin(k_p * (Xp - Xi)) / k_p
kernel *= causal_mask

# trapezoidal weights
weights = np.ones_like(xi)
weights[0] = 0.5
weights[-1] = 0.5

# Store results
results = []

for kp_sigma in kp_sigma_values:
    sigma_z = kp_sigma / k_p

    # Drive beam
    n_b = n_b0 * np.exp(-xi**2 / (2 * sigma_z**2))

    # Source term
    S = (e * n_b0 / epsilon_0) * (xi / sigma_z**2) * np.exp(-xi**2 / (2 * sigma_z**2))

    # Solve
    E_z = kernel @ (S * weights) * dxi  # [V/m]

    # Calculate analytical E_max
    E_max_analytic = (
        (e * n_b0 / epsilon_0)
        * np.sqrt(2 * np.pi)
        * sigma_z
        * np.exp(-0.5 * (k_p * sigma_z)**2)
    )

    results.append({
        "kp_sigma": kp_sigma,
        "sigma_z": sigma_z,
        "n_b": n_b,
        "E_z": E_z,
        "E_max_analytic": E_max_analytic,
    })

    print(
        f"k_p*sigma_z = {kp_sigma:.2f}, "
        f"sigma_z = {sigma_z*1e6:.2f} microns, "
        f"E_max(analytic) = {E_max_analytic:.3e} V/m"
    )

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Top: beam densities
for i, res in enumerate(results):
    color = colors[i % len(colors)]
    ax1.plot(
        xi * 1e6,
        res["n_b"],
        color=color,
        label=rf"$k_p\sigma_z={res['kp_sigma']}$"
    )

ax1.set_ylabel(r"$n_b(\xi)$ [m$^{-3}$]")
ax1.set_title(
    "Gaussian drive beams and corresponding plasma wakefields\n"
    + rf"$n_0={n_0:.2e}\,\mathrm{{m}}^{{-3}}$, "
    + rf"$n_{{b0}}={n_b0:.2e}\,\mathrm{{m}}^{{-3}}$, "
    + rf"$\lambda_p={lambda_p*1e6:.1f}\,\mu\mathrm{{m}}$"
)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Bottom: wakefields + analytic amplitudes
for i, res in enumerate(results):
    color = colors[i % len(colors)]

    ax2.plot(
        xi * 1e6,
        res["E_z"],
        color=color,
        label=rf"$k_p\sigma_z={res['kp_sigma']}$"
    )

    ax2.axhline(
        res["E_max_analytic"],
        color=color,
        linestyle=":",
        linewidth=1.5,
        alpha=0.9
    )
    ax2.axhline(
        -res["E_max_analytic"],
        color=color,
        linestyle=":",
        linewidth=1.5,
        alpha=0.9
    )

ax2.set_xlabel(r"$\xi$ [$\mu$m]")
ax2.set_ylabel(r"$E_z(\xi)$ [V/m]")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()