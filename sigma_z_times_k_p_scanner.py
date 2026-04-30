import numpy as np
import matplotlib.pyplot as plt

# Physical Constants
e = 1.602176634e-19
epsilon_0 = 8.8541878128e-12
m_e = 9.1093837015e-31
c = 299792458.0

# User Inputs
n_0 = 1.0e22      # plasma density [m^-3]
n_b0 = 2.0e21     # beam peak density [m^-3]

# Scan range in the dimensionless parameter k_p * sigma_z
kp_sigma_min = 0.2
kp_sigma_max = 3.0
nscan = 40

# Numerical domain
xi_min = -2000e-6
xi_max =  500e-6
npts = 2**13

# Region where we measure the trailing wake amplitude
tail_cut_factor = 4.0

# Derived Parameters
omega_p = np.sqrt(n_0 * e**2 / (m_e * epsilon_0))
k_p = omega_p / c
lambda_p = 2 * np.pi / k_p

print(f"lambda_p = {lambda_p*1e6:.2f} microns")
print(f"k_p = {k_p:.4e} 1/m")

# Gridding STuff
xi = np.linspace(xi_min, xi_max, npts)
dxi = xi[1] - xi[0]

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

# Scan
kp_sigma_vals = np.linspace(kp_sigma_min, kp_sigma_max, nscan)

Emax_num_vals = []
Emax_analytic_vals = []

for kp_sigma in kp_sigma_vals:
    sigma_z = kp_sigma / k_p

    # Drive beam
    n_b = n_b0 * np.exp(-xi**2 / (2 * sigma_z**2))

    # Source term
    S = (e * n_b0 / epsilon_0) * (xi / sigma_z**2) * np.exp(-xi**2 / (2 * sigma_z**2))

    # Numerical wake
    E_z = kernel @ (S * weights) * dxi

    # Analytic amplitude
    E_max_analytic = (
        (e * n_b0 / epsilon_0)
        * np.sqrt(2 * np.pi)
        * sigma_z
        * np.exp(-0.5 * (k_p * sigma_z)**2)
    )

    # Numerical trailing amplitude
    tail_mask = xi < (-tail_cut_factor * sigma_z)

    if np.any(tail_mask):
        E_max_num = np.max(np.abs(E_z[tail_mask]))
    else:
        E_max_num = np.nan

    Emax_num_vals.append(E_max_num)
    Emax_analytic_vals.append(E_max_analytic)

# Convert to arrays
Emax_num_vals = np.array(Emax_num_vals)
Emax_analytic_vals = np.array(Emax_analytic_vals)

# Plotting Stuff
plt.figure(figsize=(8, 5))
plt.plot(kp_sigma_vals, Emax_num_vals, label="Numerical $E_{\\max}$", linewidth=2)
plt.plot(kp_sigma_vals, Emax_analytic_vals, linestyle=":", linewidth=2,
         label="Analytic $E_{\\max}$")

plt.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label=r"$k_p \sigma_z = 1$")
plt.xlabel(r"$k_p \sigma_z$")
plt.ylabel(r"$E_{\max}$ [V/m]")
plt.title("Trailing wake amplitude vs bunch length")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()