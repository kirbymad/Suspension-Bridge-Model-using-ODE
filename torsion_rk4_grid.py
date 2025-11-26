import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- Parameters ----------
lam = 0.02              # forcing amplitude λ
mu  = 1.4               # forcing frequency μ
T   = 2 * math.pi / mu  # one forcing period
h   = 0.001             # RK4 time step
N_PERIODS = 40          # number of periods to let the solution stabilize

STEP = 0.1              # grid step for c and d
N_POINTS = 11           # 0, 0.1, ..., 1.0  → 11 points

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
            print(f"c = {c:.1f}, d = {d:.1f}, E(c,d) = {E_val:.4e}")

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

# ---------- Plotting: error grid ----------
def plot_error_grid(E_matrix, filename="torsion_error_grid.png"):
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

# ---------- Plotting: torsional motion vs time ----------
def plot_torsion_time_series(c0, d0, n_periods_plot=1, filename="torsion_time_series.png"):
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
    plt.plot(t_vals, theta_vals)
    plt.xlabel("Time")
    plt.ylabel("Torsional Motion in Radians")
    plt.title(f"Torsional motion, c={c0:.2f}, d={d0:.2f}")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ---------- Main ----------
if __name__ == "__main__":
    # 1. Run the grid search and get the matrix + best (c,d)
    E_matrix, best_pair, best_E = grid_search()

    # 2. Heatmap of the grid (analog of the “grid image” in the paper)
    plot_error_grid(E_matrix, filename="torsion_error_grid.png")

    # 3. Time–series plot for the best (c,d) over one period (like Fig. 4 style)
    c_best, d_best = best_pair
    plot_torsion_time_series(c_best, d_best,
                             n_periods_plot=1,
                             filename="torsion_time_series_best.png")