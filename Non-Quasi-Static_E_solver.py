import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Physical Constants
e = 1.602176634e-19
epsilon_0 = 8.8541878128e-12
m_e = 9.1093837015e-31
c = 299792458.0

# Input Parameters
n_0 = 1.0e22      # plasma density [m^-3]
n_b0 = 2.0e21     # beam peak density [m^-3]
sigma_z = 53e-6   # beam length [m]

# spatial domain
z_min = -1500e-6
z_max =  1500e-6
nz = 2**10

# time domain
t_min = -5e-12
t_max =  5e-12
nt = 2**10

# beam center at t = 0
z0 = 0.0

# animation settings
frame_step = 20
interval = 40  # ms

# Derived Parameters
omega_p = np.sqrt(n_0 * e**2 / (m_e * epsilon_0))
lambda_p = 2 * np.pi * c / omega_p
T_p = 2 * np.pi / omega_p
k_p = omega_p / c

print(f"omega_p = {omega_p:.3e} rad/s")
print(f"T_p = {T_p*1e12:.3f} ps")
print(f"lambda_p = {lambda_p*1e6:.2f} microns")
print(f"k_p sigma_z = {k_p*sigma_z:.3f}")

# Gridding Stuff
z = np.linspace(z_min, z_max, nz)
dz = z[1] - z[0]

t = np.linspace(t_min, t_max, nt)
dt = t[1] - t[0]

print(f"dt = {dt*1e15:.3f} fs")
print(f"omega_p * dt = {omega_p * dt:.3f}")

# Gaussian drive beam, moving at c
Z = z[None, :]
T = t[:, None]

n_b = n_b0 * np.exp(-((Z - z0 - c*T)**2) / (2 * sigma_z**2))

# Greens Function
Tm = t[:, None]
Tp = t[None, :]

causal_mask = (Tp <= Tm).astype(float)
time_kernel = np.sin(omega_p * (Tm - Tp)) * causal_mask

# trapezoidal weights in time
w_t = np.ones(nt)
w_t[0] = 0.5
w_t[-1] = 0.5

n1 = -omega_p * (time_kernel @ (n_b * w_t[:, None])) * dt

# Calculate electric field from Poisson
rhs = -(e / epsilon_0) * (n1 + n_b)
E = np.zeros_like(rhs)

for m in range(nt):
    E[m, 1:] = np.cumsum(0.5 * (rhs[m, 1:] + rhs[m, :-1]) * dz)

E = E - E.mean(axis=1, keepdims=True)

# Animating
z_um = z * 1e6
t_ps = t * 1e12
frames = np.arange(0, nt, frame_step)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

nb_max = 1.05 * np.max(n_b)
n1_min, n1_max = 1.05 * np.min(n1), 1.05 * np.max(n1)
E_min, E_max = 1.05 * np.min(E), 1.05 * np.max(E)

line_nb, = axes[0].plot([], [], color='blue', lw=2)
line_n1, = axes[1].plot([], [], color='purple', lw=2)
line_E,  = axes[2].plot([], [], color='red', lw=2)

beam_line0 = axes[0].axvline(0, color='black', linestyle='--', alpha=0.4)
beam_line1 = axes[1].axvline(0, color='black', linestyle='--', alpha=0.4)
beam_line2 = axes[2].axvline(0, color='black', linestyle='--', alpha=0.4)

time_text = axes[0].text(
    0.02, 0.90, "", transform=axes[0].transAxes, fontsize=11,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)

axes[0].set_ylabel(r"$n_b(z,t)$ [m$^{-3}$]")
axes[0].set_ylim(0, nb_max)
axes[0].set_title("Non-quasi-static linear plasma response")
axes[0].grid(True, alpha=0.3)

axes[1].set_ylabel(r"$n_1(z,t)$ [m$^{-3}$]")
axes[1].set_ylim(n1_min, n1_max)
axes[1].grid(True, alpha=0.3)

axes[2].set_ylabel(r"$E(z,t)$ [V/m]")
axes[2].set_xlabel(r"$z$ [$\mu$m]")
axes[2].set_ylim(E_min, E_max)
axes[2].grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlim(z_um[0], z_um[-1])

def init():
    line_nb.set_data([], [])
    line_n1.set_data([], [])
    line_E.set_data([], [])
    beam_line0.set_xdata([np.nan, np.nan])
    beam_line1.set_xdata([np.nan, np.nan])
    beam_line2.set_xdata([np.nan, np.nan])
    time_text.set_text("")

def update(frame_idx):
    m = frames[frame_idx]

    line_nb.set_data(z_um, n_b[m])
    line_n1.set_data(z_um, n1[m])
    line_E.set_data(z_um, E[m])

    beam_center_um = (z0 + c * t[m]) * 1e6
    beam_line0.set_xdata([beam_center_um, beam_center_um])
    beam_line1.set_xdata([beam_center_um, beam_center_um])
    beam_line2.set_xdata([beam_center_um, beam_center_um])

    time_text.set_text(
        rf"$t={t_ps[m]:.3f}\,\mathrm{{ps}}$"
        + "\n"
        + rf"$k_p\sigma_z={k_p*sigma_z:.3f}$"
    )

anim = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    init_func=init,
    interval=interval,
    blit=False,
    repeat=True
)
from matplotlib.animation import PillowWriter
anim.save("plasma_animation.gif", writer=PillowWriter(fps=25)) # Save 

plt.tight_layout()

plt.show()
_ = anim