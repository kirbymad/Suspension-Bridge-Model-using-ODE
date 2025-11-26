import math

# ---------- Parameters ----------
lam = 0.02            # forcing amplitude λ
mu  = 1.4             # forcing frequency μ
T   = 2 * math.pi / mu   # one forcing period
h   = 0.001           # RK4 time step

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
N_PERIODS = 20  # number of periods to let the solution stabilize

# E(c,d) = (c - y(T))^2 + (d - y'(T))^2
def error_E(c, d):
    t = 0.0
    theta = c
    omega = d

    # integrate from t=0 to t=T using RK4
    T_final = T + N_PERIODS * T
    while t < T_final:
        theta, omega = rk4_step(t, theta, omega, h)
        t += h

    E = (c - theta)**2 + (d - omega)**2
    return E

# ---------- Grid search over c,d in {0, 0.2, ..., 1.0} ----------
def grid_search():
    c = 0.0
    best_E = None
    best_pair = None

    while c <= 1.0 + 1e-9:
        d = 0.0
        while d <= 1.0 + 1e-9:
            E_val = error_E(c, d)
            print(f"c = {c:.1f}, d = {d:.1f}, E(c,d) = {E_val:.4e}")
            if best_E is None or E_val < best_E:
                best_E = E_val
                best_pair = (c, d)
            d += 0.2
        c += 0.2

    print("\nBest grid point:")
    print(f"(c, d) = ({best_pair[0]:.1f}, {best_pair[1]:.1f}),  E = {best_E:.4e}")

if __name__ == "__main__":
    grid_search()