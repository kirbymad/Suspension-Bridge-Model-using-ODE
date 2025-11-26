import math
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import argparse
from datetime import datetime

# ---------- Parameters ----------
lam = 0.02              # forcing amplitude λ
mu  = 1.4               # forcing frequency μ
T   = 2 * math.pi / mu  # one forcing period
h   = 0.001             # RK4 time step

N_PERIODS_EXPLORE = 40   # coarse/fast exploration (grid search)
N_PERIODS_BIG     = 400  # paper-scale long stabilization run
N_PERIODS = N_PERIODS_EXPLORE  # default value used by error_E/grid search

STEP = 0.1              # grid step for c and d
N_POINTS = 11           # 0, 0.1, ..., 1.0  → 11 points
OUTPUT_DIR = "results"


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def output_path(filename):
    return os.path.join(OUTPUT_DIR, filename)

# ---------- Right-hand side of the torsional system ----------
# theta' = omega
# omega' = -0.01*omega - 2.4*sin(theta) + lam*sin(mu*t)
def torsion_rhs(t, theta, omega):
    dtheta = omega
    domega = -0.01 * omega - 2.4 * math.sin(theta) + lam * math.sin(mu * t)
    return dtheta, domega

# ---------- One RK4 step ----------
def rk4_step(t, theta, omega, h):
    k1_theta, k1_omega = torsion_rhs(t, theta, omega)
    k2_theta, k2_omega = torsion_rhs(
        t + 0.5*h,
        theta + 0.5*h*k1_theta,
        omega + 0.5*h*k1_omega
    )
    k3_theta, k3_omega = torsion_rhs(
        t + 0.5*h,
        theta + 0.5*h*k2_theta,
        omega + 0.5*h*k2_omega
    )
    k4_theta, k4_omega = torsion_rhs(
        t + h,
        theta + h*k3_theta,
        omega + h*k3_omega
    )

    theta_next = theta + (h/6.0) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
    omega_next = omega + (h/6.0) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
    return theta_next, omega_next

# ---------- Vertical model (bridge roadway) ----------
# parameters for the vertical restoring force
a_vert = 17.0   # coefficient for y^+
b_vert = 1.0    # coefficient for y^-

def vertical_rhs(t, y, yp):
    """
    Right-hand side for the vertical model:
    y'' + 0.01 y' + a y^+ - b y^- = 10 + λ sin(μ t)
    """
    if y > 0:
        y_plus = y
        y_minus = 0.0
    elif y < 0:
        y_plus = 0.0
        y_minus = -y   # y^- = max(-y, 0)
    else:
        y_plus = 0.0
        y_minus = 0.0

    spring = a_vert * y_plus - b_vert * y_minus
    ypp = -0.01 * yp - spring + 10.0 + lam * math.sin(mu * t)
    return yp, ypp

def rk4_vertical_step(t, y, yp, h):
    k1_y,  k1_yp  = vertical_rhs(t, y, yp)
    k2_y,  k2_yp  = vertical_rhs(t + 0.5*h, y + 0.5*h*k1_y,  yp + 0.5*h*k1_yp)
    k3_y,  k3_yp  = vertical_rhs(t + 0.5*h, y + 0.5*h*k2_y,  yp + 0.5*h*k2_yp)
    k4_y,  k4_yp  = vertical_rhs(t + h,     y + h*k3_y,      yp + h*k3_yp)

    y_next  = y  + (h/6.0) * (k1_y  + 2*k2_y  + 2*k3_y  + k4_y)
    yp_next = yp + (h/6.0) * (k1_yp + 2*k2_yp + 2*k3_yp + k4_yp)
    return y_next, yp_next

# ---------- Error function E(c,d) ----------
# E(c,d) = (c - y(NT))^2 + (d - y'(NT))^2
def error_E(c, d):
    t = 0.0
    theta = c
    omega = d

    t_final = N_PERIODS * T   # integrate over many periods
    while t < t_final:
        theta, omega = rk4_step(t, theta, omega, h)
        t += h

    E = (c - theta)**2 + (d - omega)**2
    return E

# ---------- Grid search over c,d in {0, 0.1, ..., 1.0} with 11×11 matrix ----------
def grid_search():
    # 11×11 matrix for the error values E(c,d)
    E_matrix = [[0.0 for _ in range(N_POINTS)] for _ in range(N_POINTS)]

    best_E = None
    best_pair = None

    c = 0.0
    i = 0   # row index for matrix (for c)

    while c <= 1.0 + 1e-9 and i < N_POINTS:
        d = 0.0
        j = 0   # column index for matrix (for d)

        while d <= 1.0 + 1e-9 and j < N_POINTS:
            # compute error at this grid point
            E_val = error_E(c, d)

            # store E(c,d) in the matrix at (i,j)
            E_matrix[i][j] = E_val

            # print like in class
            print(f"c = {c:.4f}, d = {d:.4f}, E(c,d) = {E_val:.4e}")

            # track the best (smallest) error
            if best_E is None or E_val < best_E:
                best_E = E_val
                best_pair = (c, d)

            # move to next column
            d += STEP
            j += 1

        # move to next row
        c += STEP
        i += 1

    # print the full 11×11 error matrix (for the console)
    print("\nError Matrix (11×11):")
    for row in E_matrix:
        print([f"{val:.4e}" for val in row])

    # print the best grid point
    print("\nBest grid point:")
    print(f"(c, d) = ({best_pair[0]:.1f}, {best_pair[1]:.1f}),  E = {best_E:.4e}")

    return E_matrix, best_pair, best_E

# ---------- Plotting: error grid (heatmap) ----------
def plot_error_grid(E_matrix, filename=None):
    if filename is None:
        filename = output_path("torsion_error_grid.png")
    E = np.array(E_matrix)

    vals = np.linspace(0.0, 1.0, N_POINTS)
    C, D = np.meshgrid(vals, vals)

    plt.figure(figsize=(5, 4))
    im = plt.pcolormesh(C, D, E, shading='auto')
    plt.colorbar(im, label="E(c,d)")
    plt.xlabel("c")
    plt.ylabel("d")
    plt.title("Error grid for torsional model")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ---------- Plotting: 3D error surface ----------
def plot_error_surface_3d(E_matrix, filename=None):
    """
    Create a 3D surface plot of the error function E(c,d).
    """
    if filename is None:
        filename = output_path("torsion_error_surface_3d.png")
    E = np.array(E_matrix)
    
    # Get the grid values (c and d from 0.0 to 1.0 with STEP spacing)
    vals = np.linspace(0.0, 1.0, N_POINTS)
    # Use 'ij' indexing so C[i,j] = vals[i] and D[i,j] = vals[j]
    C, D = np.meshgrid(vals, vals, indexing='ij')
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(C, D, E,
                          cmap='viridis',
                          linewidth=0,
                          antialiased=True,
                          alpha=0.9)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label="E(c,d)")
    
    # Labels and title
    ax.set_xlabel("c (initial angle)", fontsize=12)
    ax.set_ylabel("d (initial angular velocity)", fontsize=12)
    ax.set_zlabel("E(c,d)", fontsize=12)
    ax.set_title("3D Error Surface for Torsional Model", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ---------- Plotting: torsional motion vs time (many periods) ----------
def plot_torsion_time_series(c0, d0, n_periods_plot=1, filename=None):
    t = 0.0
    theta = c0
    omega = d0

    t_final = n_periods_plot * T

    t_vals = []
    theta_vals = []

    while t < t_final:
        t_vals.append(t)
        theta_vals.append(theta)
        theta, omega = rk4_step(t, theta, omega, h)
        t += h

    plt.figure(figsize=(5, 4))
    plt.plot(t_vals, theta_vals, color='black', linewidth=1)
    plt.xlabel("Time")
    plt.ylabel("Torsional Motion in Radians")
    plt.title(f"Torsional motion, c={c0:.2f}, d={d0:.2f}")
    plt.tight_layout()
    if filename is None:
        filename = output_path("torsion_time_series.png")
    plt.savefig(filename, dpi=300)
    plt.close()

# ---------- Plotting: vertical motion vs time (many periods) ----------
def plot_vertical_time_series(y0, yp0, n_periods_plot=3,
                              filename=None):
    t = 0.0
    y = y0
    yp = yp0

    t_final = n_periods_plot * T

    t_vals = []
    y_vals = []

    while t < t_final:
        t_vals.append(t)
        y_vals.append(y)
        y, yp = rk4_vertical_step(t, y, yp, h)
        t += h

    plt.figure(figsize=(5,4))
    plt.plot(t_vals, y_vals, color='black', linewidth=1.1)
    plt.xlabel("Time")
    plt.ylabel("Vertical Displacement")
    plt.title(f"Vertical motion, y(0)={y0:.2f}, y'(0)={yp0:.2f}")
    plt.tight_layout()
    if filename is None:
        filename = output_path("vertical_time_series.png")
    plt.savefig(filename, dpi=300)
    plt.close()

# ---------- Plotting: vertical one-period plot with fixed axes ----------
def plot_vertical_one_period(y0, yp0, filename=None,
                             ymin=-0.5, ymax=2.0):
    """
    Plot vertical motion y(t) over exactly one forcing period T,
    with fixed axis ranges so multiple plots are comparable.
    """
    t = 0.0
    y = y0
    yp = yp0

    t_vals = []
    y_vals = []

    t_final = T  # one forcing period

    while t < t_final:
        t_vals.append(t)
        y_vals.append(y)
        y, yp = rk4_vertical_step(t, y, yp, h)
        t += h

    plt.figure(figsize=(4.8, 3.6))
    plt.plot(t_vals, y_vals, color='black', linewidth=1.1)

    # FIXED axes so all three vertical plots match
    plt.xlim(0, T)
    plt.ylim(ymin, ymax)

    plt.xlabel("One Period in Time")
    plt.ylabel("Vertical Displacement")
    plt.tight_layout()
    if filename is None:
        filename = output_path("vertical_one_period.png")
    plt.savefig(filename, dpi=300)
    plt.close()

# ---------- NEW: Figure-3-style one-period plot ----------
def plot_one_period(c0, d0, filename=None,
                    ymin=-1.0, ymax=1.0):
    """
    Plot torsional motion over exactly one forcing period T,
    with fixed x-axis and y-axis ranges for consistency.
    """
    t = 0.0
    theta = c0
    omega = d0

    t_vals = []
    theta_vals = []

    t_final = T  # exactly one period

    while t < t_final:
        t_vals.append(t)
        theta_vals.append(theta)
        theta, omega = rk4_step(t, theta, omega, h)
        t += h

    plt.figure(figsize=(4.5, 3.5))
    plt.plot(t_vals, theta_vals, color='black', linewidth=1)

    # FIXED AXIS RANGES
    plt.xlim(0, T)
    plt.ylim(ymin, ymax)

    plt.xlabel("One Period in Time")
    plt.ylabel("Torsional Displacement")
    plt.title(f"c = {c0:.2f}, d = {d0:.2f}")
    plt.tight_layout()
    if filename is None:
        filename = output_path("torsion_period.png")
    plt.savefig(filename, dpi=300)
    plt.close()

def local_grid_search(c0, d0, dc, dd, step, label):
    """
    Run a torsional grid search in a small rectangle around (c0, d0):
    c ∈ [c0-dc, c0+dc], d ∈ [d0-dd, d0+dd] with given step.
    Saves a heatmap with the label in the filename and returns
    the matrix and best (c,d).
    """
    c_min = c0 - dc
    c_max = c0 + dc
    d_min = d0 - dd
    d_max = d0 + dd
    # build lists of c and d values
    c_values = []
    x = c_min
    while x <= c_max + 1e-9:
        c_values.append(x)
        x += step

    d_values = []
    y = d_min
    while y <= d_max + 1e-9:
        d_values.append(y)
        y += step

    n_c = len(c_values)
    n_d = len(d_values)

    # matrix for E(c,d)
    E_matrix = [[0.0 for _ in range(n_d)] for _ in range(n_c)]

    best_E = None
    best_pair = None

    print(f"\n=== Local grid search around (c0={c0}, d0={d0}), label = {label} ===")

    for i, c in enumerate(c_values):
        for j, d in enumerate(d_values):
            E_val = error_E(c, d)
            E_matrix[i][j] = E_val
            print(f"c = {c:.3f}, d = {d:.3f}, E(c,d) = {E_val:.4e}")
            if best_E is None or E_val < best_E:
                best_E = E_val
                best_pair = (c, d)

    # convert to numpy array and make a heatmap
    E_np = np.array(E_matrix)
    C, D = np.meshgrid(d_values, c_values)  # note order

    plt.figure(figsize=(5, 4))
    im = plt.pcolormesh(C, D, E_np, shading='auto')
    plt.colorbar(im, label="E(c,d)")
    plt.xlabel("d (initial angular velocity)")
    plt.ylabel("c (initial angle)")
    plt.title(f"Torsional error grid near (c0={c0:.2f}, d0={d0:.2f})")
    plt.tight_layout()
    plt.savefig(f"torsion_error_grid_{label}.png", dpi=300)
    plt.close()

    print(f"\nBest near {label}: (c, d) = ({best_pair[0]:.4f}, {best_pair[1]:.4f}), E = {best_E:.4e}")
    return E_matrix, best_pair, best_E


def evaluate_pairs_with_periods(pairs, n_periods, label=""):
    """
    Recompute E(c,d) for selected pairs using a different N_PERIODS value.
    """
    global N_PERIODS
    old_periods = N_PERIODS
    N_PERIODS = n_periods

    print(f"\n=== Evaluating selected pairs with N_PERIODS = {n_periods} ({label}) ===")
    results = {}
    try:
        for name, (c_val, d_val) in pairs.items():
            E_val = error_E(c_val, d_val)
            results[name] = E_val
            print(f"{name}: (c={c_val:.4f}, d={d_val:.4f}) -> E = {E_val:.6e}")
    finally:
        N_PERIODS = old_periods

    return results

# ---------- Output capture class ----------
class Tee:
    """Class to write output to both console and file"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

# ---------- Main ----------
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='MATH333 Final Project - Torsional Model Analysis')
    parser.add_argument('--3d-only', dest='only_3d', action='store_true', 
                       help='Only generate the 3D surface plot (skip all other plots)')
    parser.add_argument('--step', type=float, default=None,
                        help='Grid step for c and d (e.g., 0.00125 for N≈800)')
    parser.add_argument('--n-points', type=int, default=None,
                        help='Number of grid points per axis (e.g., 801 for N≈800)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory where all outputs will be saved (default: results)')
    parser.add_argument('--n-periods', type=int, default=None,
                        help='Number of forcing periods to use for the main grid search (default: 40)')
    args = parser.parse_args()

    def apply_cli_overrides(parsed_args):
        global N_POINTS, STEP, OUTPUT_DIR, N_PERIODS

        if parsed_args.n_points is not None:
            if parsed_args.n_points < 2:
                raise ValueError("n-points must be at least 2")
            N_POINTS = parsed_args.n_points
            STEP = 1.0 / (N_POINTS - 1)

        if parsed_args.step is not None:
            if parsed_args.n_points is not None:
                computed = 1.0 / (parsed_args.n_points - 1)
                if abs(computed - parsed_args.step) > 1e-9:
                    print(f"Warning: provided step ({parsed_args.step}) does not match 1/(n_points-1) ({computed}). Using step={parsed_args.step}.")
            STEP = parsed_args.step
            if parsed_args.n_points is None:
                N_POINTS = int(round(1.0 / STEP)) + 1

        if parsed_args.output_dir is not None:
            OUTPUT_DIR = parsed_args.output_dir

        if parsed_args.n_periods is not None:
            if parsed_args.n_periods < 1:
                raise ValueError("n-periods must be positive")
            N_PERIODS = parsed_args.n_periods

    apply_cli_overrides(args)
    ensure_output_dir(OUTPUT_DIR)
    
    # If --3d-only flag is set, only run grid search and 3D plot
    if args.only_3d:
        print("Running in 3D-only mode...")
        E_matrix, best_pair, best_E = grid_search()
        plot_error_surface_3d(E_matrix, filename="torsion_error_surface_3d.png")
        print("3D surface plot saved as: torsion_error_surface_3d.png")
        sys.exit(0)
    
    # Set up output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = output_path(f"output_{timestamp}.txt")
    
    # Redirect stdout to both console and file
    original_stdout = sys.stdout
    with open(output_filename, 'w', encoding='utf-8') as f:
        sys.stdout = Tee(original_stdout, f)
        
        print(f"=== MATH333 Final Project Output ===")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output file: {output_filename}\n")
        
        try:
            # 1. Run the grid search and get the matrix + best (c,d)
            E_matrix, best_pair, best_E = grid_search()

            # 2. Heatmap of the grid (analog of the "grid image")
            plot_error_grid(E_matrix, filename=output_path("torsion_error_grid.png"))
            
            # 2b. 3D surface plot of the error
            plot_error_surface_3d(E_matrix, filename=output_path("torsion_error_surface_3d.png"))

            # 3. Local grid searches around three centers for refined solutions
            # Small center: (0.0, 0.1)
            _, best_small, _ = local_grid_search(0.0, 0.1, 0.1, 0.1, 0.01, "small")
            c_small, d_small = best_small

            # Medium center: (0.3, 0.3)
            _, best_medium, _ = local_grid_search(0.3, 0.3, 0.1, 0.1, 0.01, "medium")
            c_medium, d_medium = best_medium

            # Large center: (0.7, 0.2)
            _, best_large, _ = local_grid_search(0.7, 0.2, 0.1, 0.1, 0.01, "large")
            c_large, d_large = best_large

            # 3b. Evaluate selected pairs with a long stabilization run
            selected_pairs = {
                "best_grid": best_pair,
                "local_small": best_small,
                "local_medium": best_medium,
                "local_large": best_large,
            }
            evaluate_pairs_with_periods(selected_pairs,
                                        N_PERIODS_BIG,
                                        label="paper-scale")

            # 4. Time–series plot for the best (c,d) from main grid search
            c_best, d_best = best_pair
            plot_torsion_time_series(c_best, d_best,
                                     n_periods_plot=3,
                                     filename=output_path("torsion_time_series_best.png"))

            # 5. Figure-3-style one-period plots using refined values from local searches
            # Small stable (refined from local search around (0.0, 0.1))
            plot_one_period(c_small, d_small, output_path("fig3_small_stable.png"),
                            ymin=-1.0, ymax=1.0)

            # Medium/large unstable (refined from local search around (0.3, 0.3))
            plot_one_period(c_medium, d_medium, output_path("fig3_large_unstable.png"),
                            ymin=-1.0, ymax=1.0)

            # Large stable (refined from local search around (0.7, 0.2))
            plot_one_period(c_large, d_large, output_path("fig3_large_stable.png"),
                            ymin=-1.0, ymax=1.0)

            # --- VERTICAL MODEL: Figure-3-style plots (small / large unstable / large stable) ---

            # Small-amplitude vertical solution (paper's "intuitively obvious" one)
            y0_small  = 0.584460095
            yp0_small = 0.401083422
            plot_vertical_one_period(y0_small, yp0_small,
                                     output_path("vert_small_stable.png"),
                                     ymin=-0.5, ymax=2.0)

            # Large-amplitude unstable vertical solution
            y0_unst  = 0.344135876
            yp0_unst = 2.77765496
            plot_vertical_one_period(y0_unst, yp0_unst,
                                     output_path("vert_large_unstable.png"),
                                     ymin=-0.5, ymax=2.0)

            # Large-amplitude stable vertical solution
            y0_large  = 0.312999338
            yp0_large = -2.87405731
            plot_vertical_one_period(y0_large, yp0_large,
                                     output_path("vert_large_stable.png"),
                                     ymin=-0.5, ymax=2.0)
        
        finally:
            # Restore original stdout
            sys.stdout = original_stdout
    
    print(f"\nOutput saved to: {output_filename}")