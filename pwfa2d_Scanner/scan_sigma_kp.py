import os
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import yt

# Commands used to run warpx
WARPX_EXE = r"C:\Users\Jason\miniforge3\envs\warpx\Library\bin\warpx.2d.NOMPI.OMP.DP.PDP.OPMD.FFT.EB.QED.exe"
INPUT_FILE = "inputs_test_2d_plasma_acceleration_boosted"

MAX_STEP = 1000
DIAG_INTERVAL = 1000

DENSITY = 1.0e23  # m^-3

SIGMA_KP_VALUES = np.arange(0.10, 1.1001, 0.05)

RESULTS_CSV = "sigma_kp_scan_results.csv"
RESULTS_PNG = "sigma_kp_vs_Emax.png"

# Emax search box 
Z_MIN_CM = 0.10
Z_MAX_CM = 0.25
X_WINDOW_UM = 10.0

# Constants
e = 1.602176634e-19
eps0 = 8.8541878128e-12
me = 9.1093837015e-31
c = 299792458.0

omega_p = np.sqrt(DENSITY * e**2 / (eps0 * me))
k_p = omega_p / c
lambda_p = 2.0 * np.pi / k_p

print(f"omega_p  = {omega_p:.6e} rad/s")
print(f"k_p      = {k_p:.6e} 1/m")
print(f"lambda_p = {lambda_p * 1e6:.6f} um")

# Helpers
def values(arr, unit=None):
    try:
        if unit is None:
            return np.asarray(arr.v)
        return np.asarray(arr.to(unit).v)
    except Exception:
        try:
            if unit is None:
                return np.asarray(arr)
            return np.asarray(arr.to_value(unit))
        except Exception:
            return np.asarray(arr)


def reduce_field_to_2d(arr):
    arr = np.squeeze(np.asarray(arr))

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        for ax, n in enumerate(arr.shape):
            if n == 1:
                return np.take(arr, 0, axis=ax)

    raise RuntimeError(f"Could not reduce field to 2D. Shape = {arr.shape}")


def load_Ez(plotfile):
    ds = yt.load(plotfile)

    cg = ds.covering_grid(
        level=0,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions
    )

    Ez = reduce_field_to_2d(values(cg[("boxlib", "Ez")]))
    Ez_TVm = Ez / 1e12

    left = values(ds.domain_left_edge, "m")
    right = values(ds.domain_right_edge, "m")

    x = np.linspace(left[0], right[0], Ez.shape[0])
    z = np.linspace(left[1], right[1], Ez.shape[1])

    return Ez_TVm, x, z


def first_wake_emax(plotfile):
    Ez, x, z = load_Ez(plotfile)

    x_mask = np.abs(x) <= X_WINDOW_UM * 1e-6
    z_mask = (z >= Z_MIN_CM * 1e-2) & (z <= Z_MAX_CM * 1e-2)

    region = Ez[np.ix_(x_mask, z_mask)]

    if region.size == 0:
        raise RuntimeError("Empty Emax search region. Check x/z window.")

    Ez_min = np.nanmin(region)
    Emax = abs(Ez_min)

    local_idx = np.unravel_index(np.nanargmin(region), region.shape)

    x_region = x[x_mask]
    z_region = z[z_mask]

    x_at_min = x_region[local_idx[0]]
    z_at_min = z_region[local_idx[1]]

    return Emax, Ez_min, x_at_min, z_at_min


# Run Scan
results = []

for sigma_kp in SIGMA_KP_VALUES:
    sigma_z = sigma_kp / k_p

    case_name = f"sigma_kp_{sigma_kp:.2f}".replace(".", "p")
    case_dir = os.path.abspath(case_name)

    print("\n" + "=" * 70)
    print(f"Running sigma_z * k_p = {sigma_kp:.2f}")
    print(f"sigma_z = {sigma_z:.6e} m = {sigma_z * 1e6:.3f} um")
    print(f"case dir = {case_dir}")

    os.makedirs(case_dir, exist_ok=True)
    shutil.copy2(INPUT_FILE, os.path.join(case_dir, INPUT_FILE))

    diags_dir = os.path.join(case_dir, "diags")
    if os.path.exists(diags_dir):
        shutil.rmtree(diags_dir)

    cmd = [
        WARPX_EXE,
        INPUT_FILE,
        f"max_step={MAX_STEP}",
        f"diag1.intervals={DIAG_INTERVAL}",
        f"driver.z_rms={sigma_z:.12e}",
    ]

    print("Command:")
    print(" ".join(cmd))

    subprocess.run(cmd, cwd=case_dir, check=True)

    plotfile = os.path.join(case_dir, "diags", "diag1001000")

    Emax, Ez_min, x_at_min, z_at_min = first_wake_emax(plotfile)

    print(f"Emax = |min(Ez)| = {Emax:.6e} TV/m")
    print(f"min(Ez)          = {Ez_min:.6e} TV/m")
    print(f"x at min         = {x_at_min * 1e6:.3f} um")
    print(f"z at min         = {z_at_min * 1e2:.6f} cm")

    results.append([
        sigma_kp,
        sigma_z,
        sigma_z * 1e6,
        Emax,
        Ez_min,
        x_at_min,
        z_at_min,
    ])


# Save CSV
results = np.array(results)

header = (
    "sigma_kp,"
    "sigma_z_m,"
    "sigma_z_um,"
    "Emax_abs_TVm,"
    "Ez_min_TVm,"
    "x_at_min_m,"
    "z_at_min_m"
)

np.savetxt(
    RESULTS_CSV,
    results,
    delimiter=",",
    header=header,
    comments=""
)

print(f"\nSaved {RESULTS_CSV}")

# Plotting Stuff
plt.figure(figsize=(7.2, 5.0), dpi=180)
plt.plot(results[:, 0], results[:, 3], marker="o")
plt.xlabel(r"$\sigma_z k_p$")
plt.ylabel(r"$E_{\max}=|\min(E_z)|$ for $0.10<z<0.25$ cm [TV/m]")
plt.title(r"First wake field vs. drive bunch length, fixed total charge")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_PNG, dpi=180)

print(f"Saved {RESULTS_PNG}")