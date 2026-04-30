import sys
import numpy as np
import matplotlib.pyplot as plt
import yt


# Settings
SHOW_DRIVER = True
SHOW_BEAM = True
SHOW_PLASMA_E = False  

DRIVER_COLOR = "red"
BEAM_COLOR = "blue"
PLASMA_E_COLOR = "green"

MAX_DRIVER_POINTS = 4000
MAX_BEAM_POINTS = 4000
MAX_PLASMA_E_POINTS = 15000

FIGSIZE = (8.3, 6.2)
DPI = 180
plotfile = sys.argv[1] if len(sys.argv) > 1 else r"diags\diag1001000"



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


def get_2d_field(ds, cg, field_basename="Ez"):

    preferred = [
        ("boxlib", field_basename),
        ("mesh", field_basename),
        ("amrex", field_basename),
    ]

    for f in preferred:
        try:
            arr = np.squeeze(values(cg[f]))
            if arr.ndim == 2:
                return f, arr
        except Exception:
            pass

    for f in ds.field_list:
        if f[-1] == field_basename:
            try:
                arr = np.squeeze(values(cg[f]))
                if arr.ndim == 2:
                    return f, arr
            except Exception:
                pass

    raise RuntimeError(
        f"Could not find a 2D field named '{field_basename}'.\n"
        f"Available fields:\n{ds.field_list}"
    )


def get_species_coords(ad, species):
    x = values(ad[(species, "particle_position_x")], "m")
    y = values(ad[(species, "particle_position_y")], "m")
    return x, y


def downsample_pair(x, y, max_points):
    n = len(x)
    if n <= max_points:
        return x, y
    stride = max(1, n // max_points)
    return x[::stride], y[::stride]


# Load data
ds = yt.load(plotfile)
ad = ds.all_data()

cg = ds.covering_grid(
    level=0,
    left_edge=ds.domain_left_edge,
    dims=ds.domain_dimensions
)

field_name, Ez = get_2d_field(ds, cg, "Ez")

left = values(ds.domain_left_edge, "m")
right = values(ds.domain_right_edge, "m")

x_um = np.linspace(left[0] * 1e6, right[0] * 1e6, Ez.shape[0])
z_cm = np.linspace(left[1] * 1e2, right[1] * 1e2, Ez.shape[1])

Ez_tvm = Ez / 1e12

lim = np.nanpercentile(np.abs(Ez_tvm), 99.5)
if not np.isfinite(lim) or lim <= 0:
    lim = np.nanmax(np.abs(Ez_tvm))
if not np.isfinite(lim) or lim <= 0:
    lim = 1.0

# Plot background field
fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

im = ax.imshow(
    Ez_tvm,
    origin="lower",
    extent=[z_cm[0], z_cm[-1], x_um[0], x_um[-1]],
    aspect="auto",
    cmap="RdBu",
    vmin=-lim,
    vmax=lim,
    interpolation="nearest"
)

# Particle overlays
if SHOW_DRIVER:
    try:
        x, y = get_species_coords(ad, "driver")
        x, y = downsample_pair(x, y, MAX_DRIVER_POINTS)
        ax.scatter(
            y * 1e2,      # cm
            x * 1e6,      # um
            s=4,
            c=DRIVER_COLOR,
            alpha=0.9,
            edgecolors="none",
            label="driver"
        )
    except Exception as e:
        print(f"Could not plot driver: {e}")

if SHOW_BEAM:
    try:
        x, y = get_species_coords(ad, "beam")
        x, y = downsample_pair(x, y, MAX_BEAM_POINTS)
        ax.scatter(
            y * 1e2,      # cm
            x * 1e6,      # um
            s=4,
            c=BEAM_COLOR,
            alpha=0.9,
            edgecolors="none",
            label="beam"
        )
    except Exception as e:
        print(f"Could not plot beam: {e}")

if SHOW_PLASMA_E:
    try:
        x, y = get_species_coords(ad, "plasma_e")
        x, y = downsample_pair(x, y, MAX_PLASMA_E_POINTS)
        ax.scatter(
            y * 1e2,      # cm
            x * 1e6,      # um
            s=0.2,
            c=PLASMA_E_COLOR,
            alpha=0.15,
            edgecolors="none",
            label="plasma_e"
        )
    except Exception as e:
        print(f"Could not plot plasma_e: {e}")


# Labels, colorbar, save
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Ez [TV/m]")

ax.set_xlabel("Propagation direction (z) [cm]")
ax.set_ylabel("Transverse direction (x) [μm]")
ax.set_title("2D PWFA: Ez with particle overlay")

if SHOW_DRIVER or SHOW_BEAM or SHOW_PLASMA_E:
    ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("pwfa_Ez_particles.png", dpi=DPI)
print("Saved pwfa_Ez_particles.png")
print(f"Used field: {field_name}")
