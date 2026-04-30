import numpy as np
import matplotlib.pyplot as plt

# Physical Constants
e = 1.602176634e-19
epsilon_0 = 8.8541878128e-12
m_e = 9.1093837015e-31
c = 299792458.0

# Input Parameters
n_0 = 1.0e22          # plasma density [m^-3]

# Reference beam used to define fixed total charge
n_b0_ref = 2.0e21     # reference peak density [m^-3]
sigma_ref = 53e-6     # reference bunch length [m]

# Scan range in k_p sigma_z
kp_sigma_min = 0.1
kp_sigma_max = 1.1
nscan = 50

# Numerical domain
xi_min = -2000e-6
xi_max = 500e-6
npts = 2**13

# Tail region used to measure numerical wake amplitude
tail_cut_factor = 4.0

# Derived Parameters
omega_p = np.sqrt(n_0 * e**2 / (m_e * epsilon_0))
k_p = omega_p / c
lambda_p = 2 * np.pi / k_p

print("lambda_p = {:.2f} microns".format(lambda_p * 1e6))
print("k_p = {:.4e} 1/m".format(k_p))
print("reference k_p sigma_ref = {:.3f}".format(k_p * sigma_ref))

# Fixed 1D total charge density factor:
N_total = n_b0_ref * np.sqrt(2 * np.pi) * sigma_ref

print("Fixed integral n_b dxi = {:.4e} m^-2".format(N_total))

# Gridding STuff
xi = np.linspace(xi_min, xi_max, npts)
dxi = xi[1] - xi[0]

# Correct Greens function
Xi = xi[:, None]
Xp = xi[None, :]

causal_mask = (Xp >= Xi).astype(float)
kernel = np.sin(k_p * (Xp - Xi)) / k_p
kernel *= causal_mask

weights = np.ones_like(xi)
weights[0] = 0.5
weights[-1] = 0.5

# Scan
kp_sigma_vals = np.linspace(kp_sigma_min, kp_sigma_max, nscan)

Emax_num_vals = []
Emax_analytic_vals = []
sigma_vals = []
nb0_vals = []

for kp_sigma in kp_sigma_vals:
    sigma_z = kp_sigma / k_p

    # Fixed total charge condition: n_b0 * sqrt(2*pi) * sigma_z = N_total
    n_b0 = N_total / (np.sqrt(2 * np.pi) * sigma_z)

    # Drive beam
    n_b = n_b0 * np.exp(-xi**2 / (2 * sigma_z**2))

    # Source term
    S = (e * n_b0 / epsilon_0) * (xi / sigma_z**2) * np.exp(
        -xi**2 / (2 * sigma_z**2)
    )

    # Numerical wake
    E_z = kernel @ (S * weights) * dxi

    # Analytic Emax with fixed total charge
    E_max_analytic = (e / epsilon_0) * N_total * np.exp(
        -0.5 * (k_p * sigma_z)**2
    )

    # Numerical trailing amplitude, measured behind the beam
    tail_mask = xi < (-tail_cut_factor * sigma_z)

    if np.any(tail_mask):
        E_max_num = np.max(np.abs(E_z[tail_mask]))
    else:
        E_max_num = np.nan

    Emax_num_vals.append(E_max_num)
    Emax_analytic_vals.append(E_max_analytic)
    sigma_vals.append(sigma_z)
    nb0_vals.append(n_b0)

Emax_num_vals = np.array(Emax_num_vals)
Emax_analytic_vals = np.array(Emax_analytic_vals)
sigma_vals = np.array(sigma_vals)
nb0_vals = np.array(nb0_vals)


# Emax vs k_p sigma_z =========================================================
plt.figure(figsize=(8, 5))

plt.plot(
    kp_sigma_vals,
    Emax_num_vals,
    label="Numerical Emax",
    linewidth=2
)

plt.plot(
    kp_sigma_vals,
    Emax_analytic_vals,
    linestyle=":",
    linewidth=2,
    label="Analytic Emax, fixed total charge"
)

plt.xlabel("k_p sigma_z")
plt.ylabel("Emax [V/m]")
plt.title("Wake amplitude vs bunch length, fixed total charge")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# peak density needed to keep charge fixed ====================================
plt.figure(figsize=(8, 5))

plt.plot(kp_sigma_vals, nb0_vals, linewidth=2)

plt.xlabel("k_p sigma_z")
plt.ylabel("n_b0 [m^-3]")
plt.title("Peak beam density required for fixed total charge")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()