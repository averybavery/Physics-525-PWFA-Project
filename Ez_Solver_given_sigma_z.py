import numpy as np
import matplotlib.pyplot as plt

# Physical constants
e = 1.602176634e-19
epsilon_0 = 8.8541878128e-12
m_e = 9.1093837015e-31
c = 299792458.0

# Input parameters
n_0 = 1.0e22      # plasma density [m^-3]
n_b0 = 2.0e21     # beam peak density [m^-3]
sigma_z = 10e-6   # bunch length [m]

xi_min = -2000e-6   
xi_max =  500e-6    
npts = 2**13

# Derived parameters
omega_p = np.sqrt(n_0 * e**2 / (m_e * epsilon_0))
k_p = omega_p / c
lambda_p = 2 * np.pi / k_p

print(f"k_p*sigma_z = {k_p * sigma_z:.3f}")
print(f"lambda_p = {lambda_p*1e6:.2f} microns")

# Gridding stuff
xi = np.linspace(xi_min, xi_max, npts)
dxi = xi[1] - xi[0]

# Drive beam
n_b = n_b0 * np.exp(-xi**2 / (2 * sigma_z**2))

# Source term
S = (e * n_b0 / epsilon_0) * (xi / sigma_z**2) * np.exp(-xi**2 / (2 * sigma_z**2))

# Correct Greens Function
Xi = xi[:, None]
Xp = xi[None, :]

causal_mask = (Xp >= Xi).astype(float)
kernel = np.sin(k_p * (Xp - Xi)) / k_p
kernel *= causal_mask

# trapezoidal weights
weights = np.ones_like(xi)
weights[0] = 0.5
weights[-1] = 0.5

E_z = kernel @ (S * weights) * dxi  # [V/m]

# analytical E_max
E_max_analytic = (
    (e * n_b0 / epsilon_0)
    * np.sqrt(2 * np.pi)
    * sigma_z
    * np.exp(-0.5 * (k_p * sigma_z)**2)
)

print(f"E_max (analytic) = {E_max_analytic:.3e} V/m")

# Plotting stuff
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

# Top: beam density
ax1.plot(xi * 1e6, n_b, color="blue")
ax1.set_ylabel(r"$n_b(\xi)$ [m$^{-3}$]")
ax1.set_title(
    "Gaussian drive beam and plasma wakefield\n"
    + rf"$n_0={n_0:.2e}$, "
    + rf"$n_{{b0}}={n_b0:.2e}$, "
    + rf"$\sigma_z={sigma_z*1e6:.1f}\,\mu m$, "
    + rf"$k_p\sigma_z={k_p*sigma_z:.2f}$"
)
ax1.grid(True, alpha=0.3)

# Bottom: wakefield
ax2.plot(xi * 1e6, E_z, color="red", label="Numerical $E_z$")

# Analytic amplitude lines
ax2.axhline(E_max_analytic, color="red", linestyle=":", linewidth=2,
            label="Analytic $E_{max}$")
ax2.axhline(-E_max_analytic, color="red", linestyle=":", linewidth=2)

ax2.set_xlabel(r"$\xi$ [$\mu$m]")
ax2.set_ylabel(r"$E_z(\xi)$ [V/m]")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()