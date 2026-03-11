# **RAMSES-Py (Relativistic Anisotropic Multi-Fluid Solver for Eigenmodes in Python)**
# is a Python-based extension of the original MATLAB version of **PDRF**,
# developed for plasma wave analysis and mode diagnosis.  
# Built upon the core framework of PDRF, RAMSES-Py introduces several new capabilities, including:

# - analysis of **polarization relations under eigenvalue perturbation**,
# - **simultaneous identification and classification of multiple wave branches**,
# - extended post-processing and diagnostic tools,
# - and example-based workflows for practical case studies.

# The goal of this project is to provide a more flexible, extensible, 
# and user-friendly platform for dispersion-relation analysis in magnetized plasmas.

# version 2.0, 2026-03-11, Xie Haoen and He Jiansen

# reference: 
# Computer Physics Communications 185 (2014) 670–675,
# Xie Huasheng,
# PDRF: A general dispersion relation solver for magnetizedmulti-fluid plasma
# doi: https://doi.org/10.1016/j.cpc.2013.10.012

import numpy as np
from scipy.linalg import eig
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment
import matplotlib as mpl
import matplotlib.pyplot as plt
from dataclasses import dataclass

'''settings, User-defined parameters'''
'''------------------------------------------------------------------------------'''
kmin = 0.05
kmax = 5
dk = 0.01
theta = 85 * np.pi / 180  
'''------------------------------------------------------------------------------'''

@dataclass
class PDRFState:
    s: int
    qs: np.ndarray
    ms: np.ndarray
    ns0: np.ndarray
    vs0x: np.ndarray
    vs0y: np.ndarray
    vs0z: np.ndarray
    csz2: np.ndarray
    csp2: np.ndarray
    epsnx: np.ndarray
    epsny: np.ndarray
    rhos0: np.ndarray
    Q0: float
    J0x: float
    J0y: float
    J0z: float
    nu: np.ndarray
    B0: float
    c2: float
    mu0: float
    epsilon0: float
    gammaTs: np.ndarray
    Psz: np.ndarray
    Psp: np.ndarray
    betasz: np.ndarray
    betasp: np.ndarray
    Deltas: np.ndarray
    vA: np.ndarray


def initial(input_file="pdrf.in") -> PDRFState:
    """
    Read the input file and initialize the state.
    pdrf.in file has 10 columns per line:
    q, m, n0, v0x, v0y, v0z, csz, csp, epsnx, epsny
    """
    data = np.loadtxt(input_file, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    s, col = data.shape
    if col != 10:
        raise ValueError("Wrong input data: pdrf.in must have 10 columns.")

    qs = data[:, 0]
    ms = data[:, 1]
    ns0 = data[:, 2]
    vs0x = data[:, 3]
    vs0y = data[:, 4]
    vs0z = data[:, 5]
    csz2 = data[:, 6] ** 2
    csp2 = data[:, 7] ** 2
    epsnx = data[:, 8]
    epsny = data[:, 9]

    rhos0 = ns0 * ms
    Q0 = np.sum(qs * ns0)
    J0x = np.sum(qs * ns0 * vs0x)
    J0y = np.sum(qs * ns0 * vs0y)
    J0z = np.sum(qs * ns0 * vs0z)

    if abs(Q0) > 1e-12:
        print("Warning: Non-neutral charged plasma!")

    # default Collision parameters 
    nue = 0.0
    nuee = 0.0
    nui = 0.0
    nuii = 0.0
    nuei = 0.0

    nu = np.zeros((s, s), dtype=float)
    for i in range(s):
        if ms[i] <= 1:
            nu[i, i] = nue
        else:
            nu[i, i] = nui

        for j in range(i + 1, s):
            if ms[i] <= 1 and ms[j] <= 1:
                nu[i, j] = nuee
                nu[j, i] = nuee
            elif ms[i] <= 1 and ms[j] > 1:
                nu[i, j] = nuei
                nu[j, i] = nuei
            else:
                nu[i, j] = nuii
                nu[j, i] = nuii

    # Normalized parameters
    B0 = 1.0
    c2 = 1.0
    epsilon0 = 1.0
    mu0 = 1.0 / (c2 * epsilon0)

    gammaTs = np.ones_like(qs)
    Psz = csz2 * rhos0 / gammaTs
    Psp = csp2 * rhos0 / gammaTs

    if B0 != 0:
        betasz = 2 * mu0 * Psz / B0**2
        betasp = 2 * mu0 * Psp / B0**2
        vA = B0 / np.sqrt(rhos0)
        Deltas = -(Psp - Psz) / B0
    else:
        betasz = np.zeros_like(qs)
        betasp = np.zeros_like(qs)
        vA = np.zeros_like(qs)
        Deltas = np.zeros_like(qs)

    return PDRFState(
        s=s,
        qs=qs,
        ms=ms,
        ns0=ns0,
        vs0x=vs0x,
        vs0y=vs0y,
        vs0z=vs0z,
        csz2=csz2,
        csp2=csp2,
        epsnx=epsnx,
        epsny=epsny,
        rhos0=rhos0,
        Q0=Q0,
        J0x=J0x,
        J0y=J0y,
        J0z=J0z,
        nu=nu,
        B0=B0,
        c2=c2,
        mu0=mu0,
        epsilon0=epsilon0,
        gammaTs=gammaTs,
        Psz=Psz,
        Psp=Psp,
        betasz=betasz,
        betasp=betasp,
        Deltas=Deltas,
        vA=vA,
    )


def pdrfsolver(
    state: PDRFState,
    kx: complex,
    kz: complex,
    rel: int = 0,
    return_eigenvectors: bool = True,
    raw_output: bool = False,
):
    """
    Main solver：
    1) Seek generalized eigenvalues M x = lambda A x
    2) frequency w = 1j * lambda

    parameters:
    ----
    raw_output:
        True: Directly return the unsorted eigenfrequencies/eigenvectors, 
              which are used for wave pattern tracking.
        False: Maintain the original output behavior of sorting by real part/imaginary part separately.
    """
    s = state.s
    n = 4 * s + 6

    M = np.zeros((n, n), dtype=complex)
    A = np.eye(n, dtype=complex)

    wcs = state.qs * state.B0 / state.ms
    ind2 = 4 * s

    for j in range(s):
        ind = 4 * j

        # dn ~ n
        M[0 + ind, 0 + ind] = -1j * (kx * state.vs0x[j] + kz * state.vs0z[j])

        # dn ~ v
        M[0 + ind, 1 + ind] = -1j * kx * state.ns0[j] - state.epsnx[j] * state.ns0[j]
        M[0 + ind, 2 + ind] = -state.epsny[j] * state.ns0[j]
        M[0 + ind, 3 + ind] = -1j * kz * state.ns0[j]

        # The relativistic correction block
        ajpq = np.eye(3, dtype=complex)
        if rel == 1:
            vj0 = np.sqrt(state.vs0x[j]**2 + state.vs0y[j]**2 + state.vs0z[j]**2)
            gammaj0 = 1.0 / np.sqrt(1.0 - vj0**2 / state.c2)

            ajpq[0, 0] = gammaj0 + gammaj0**3 * state.vs0x[j]**2 / state.c2
            ajpq[0, 1] = gammaj0**3 * state.vs0x[j] * state.vs0y[j] / state.c2
            ajpq[0, 2] = gammaj0**3 * state.vs0x[j] * state.vs0z[j] / state.c2

            ajpq[1, 0] = gammaj0**3 * state.vs0x[j] * state.vs0y[j] / state.c2
            ajpq[1, 1] = gammaj0 + gammaj0**3 * state.vs0y[j]**2 / state.c2
            ajpq[1, 2] = gammaj0**3 * state.vs0y[j] * state.vs0z[j] / state.c2

            ajpq[2, 0] = gammaj0**3 * state.vs0x[j] * state.vs0z[j] / state.c2
            ajpq[2, 1] = gammaj0**3 * state.vs0z[j] * state.vs0y[j] / state.c2
            ajpq[2, 2] = gammaj0 + gammaj0**3 * state.vs0z[j]**2 / state.c2

        A[ind + 1: ind + 4, ind + 1: ind + 4] = ajpq

        bjpq = -1j * (kx * state.vs0x[j] + kz * state.vs0z[j]) * ajpq
        M[ind + 1: ind + 4, ind + 1: ind + 4] = bjpq

        # dv ~ n & v
        M[ind + 1, ind + 0] = -1j * kx * state.csp2[j] / state.rhos0[j]
        M[ind + 3, ind + 0] = -1j * kz * state.csz2[j] / state.rhos0[j]
        M[ind + 1, ind + 2] += wcs[j]
        M[ind + 2, ind + 1] -= wcs[j]

        # dv ~ E
        M[ind + 1, ind2 + 0] = state.qs[j] / state.ms[j]
        M[ind + 2, ind2 + 1] = state.qs[j] / state.ms[j]
        M[ind + 3, ind2 + 2] = state.qs[j] / state.ms[j]

        # dv ~ B
        M[ind + 1, ind2 + 4] = -state.qs[j] / state.ms[j] * state.vs0z[j]
        M[ind + 1, ind2 + 5] = state.qs[j] / state.ms[j] * state.vs0y[j]
        M[ind + 2, ind2 + 3] = state.qs[j] / state.ms[j] * state.vs0z[j]
        M[ind + 2, ind2 + 5] = -state.qs[j] / state.ms[j] * state.vs0x[j]
        M[ind + 3, ind2 + 3] = -state.qs[j] / state.ms[j] * state.vs0y[j]
        M[ind + 3, ind2 + 4] = state.qs[j] / state.ms[j] * state.vs0x[j]

        # Anisotropic pressure correction
        M[ind + 1, ind2 + 3] = -1j * kz * state.Deltas[j] / state.rhos0[j]
        M[ind + 2, ind2 + 4] = -1j * kz * state.Deltas[j] / state.rhos0[j]
        M[ind + 3, ind2 + 3] = (
            M[ind + 3, ind2 + 3]
            - 1j * kx * 2.0 * state.Deltas[j] / state.rhos0[j]
        )

        # dE ~ n
        M[ind2 + 0, ind + 0] = -state.qs[j] * state.vs0x[j] / state.epsilon0
        M[ind2 + 1, ind + 0] = -state.qs[j] * state.vs0y[j] / state.epsilon0
        M[ind2 + 2, ind + 0] = -state.qs[j] * state.vs0z[j] / state.epsilon0

        # dE ~ v
        M[ind2 + 0, ind + 1] = -state.qs[j] * state.ns0[j] / state.epsilon0
        M[ind2 + 1, ind + 2] = -state.qs[j] * state.ns0[j] / state.epsilon0
        M[ind2 + 2, ind + 3] = -state.qs[j] * state.ns0[j] / state.epsilon0

    # dE ~ B
    M[ind2 + 0, ind2 + 4] = -1j * kz * state.c2
    M[ind2 + 1, ind2 + 3] = 1j * kz * state.c2
    M[ind2 + 1, ind2 + 5] = -1j * kx * state.c2
    M[ind2 + 2, ind2 + 4] = 1j * kx * state.c2

    # dB ~ E
    M[ind2 + 3, ind2 + 1] = 1j * kz
    M[ind2 + 4, ind2 + 0] = -1j * kz
    M[ind2 + 4, ind2 + 2] = 1j * kx
    M[ind2 + 5, ind2 + 1] = -1j * kx

    # Collision terms
    for i in range(s):
        for j in range(s):
            col = 4 * i
            row = 4 * j
            M[row + 1, col + 1] += state.nu[i, j]
            M[row + 2, col + 2] += state.nu[i, j]
            M[row + 3, col + 3] += state.nu[i, j]

    eigenvals, eigenvecs = eig(M, A)
    wtmp = 1j * eigenvals

    if raw_output:
        if return_eigenvectors:
            return M, A, wtmp, eigenvecs
        return M, A, wtmp

    inw1 = np.argsort(np.real(wtmp))[::-1]
    inw2 = np.argsort(np.imag(wtmp))[::-1]

    w1 = wtmp[inw1]
    w2 = wtmp[inw2]

    if return_eigenvectors:
        X1 = eigenvecs[:, inw1]
        X2 = eigenvecs[:, inw2]
        return M, A, w1, w2, X1, X2

    return M, A, w1, w2


def build_mode_styles(ns, nk):
    """
    Each pair of wave modes comes with a fixed set of styles:
    - color：Remain consistent across all subplots
    - marker：Even if the curves overlap, 
              it can be clearly seen through the markers how many pieces are pressed together.
    - markevery：Markers of different modes appear at different k points, avoiding complete overlap.
    """
    colors = mpl.colormaps["tab20"].resampled(ns)(np.linspace(0.0, 1.0, ns))
    markers = [
        "o", "s", "^", "v", "D", "P", "X",
        "<", ">", "h", "*", "p", "8", "d",
        "H", "+", "x"
    ]

    mark_step = max(ns + 3, nk // 10)

    styles = []
    for im in range(ns):
        styles.append({
            "color": colors[im],
            "marker": markers[im % len(markers)],
            "markevery": (im, mark_step),
        })
    return styles


def _spread_y_for_labels(y, min_sep):
    """
    Slightly widen the right end label in the y direction to avoid overlapping of the "mode" label.
    Note: Only move the label position, do not move the actual curve.
    """
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y

    order = np.argsort(y)
    ys = y[order].copy()

    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_sep:
            ys[i] = ys[i - 1] + min_sep

    ys -= (np.mean(ys) - np.mean(y))

    out = np.empty_like(ys)
    out[order] = ys
    return out


def add_mode_labels(ax, x_end, y_end, styles, prefix="m", xpad_frac=0.03, min_sep_frac=0.035):
    """
    On the right side, assign a mode number to each wave pattern.
    Even if two patterns are exactly the same on Re(omega), they can be distinguished by the two leads:
    They correspond to the same real part curve.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    dy = max(y1 - y0, 1.0)
    y_lab = _spread_y_for_labels(y_end, min_sep=min_sep_frac * dy)
    x_text = x1 + xpad_frac * (x1 - x0)

    for im, (yy, yl) in enumerate(zip(y_end, y_lab)):
        color = styles[im]["color"]
        ax.plot(
            [x_end, x_text],
            [yy, yl],
            color=color,
            linewidth=0.8,
            alpha=0.65,
            clip_on=False,
        )
        ax.text(
            x_text,
            yl,
            f"{prefix}{im:02d}",
            color=color,
            fontsize=8,
            va="center",
            ha="left",
            clip_on=False,
        )

    ax.set_xlim(x0, x1 + 0.18 * (x1 - x0))

def normalize_columns(X, eps=1e-14):
    X = np.asarray(X, dtype=complex)
    norms = np.linalg.norm(X, axis=0)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms


def predict_next_omega(k_hist, omega_hist, k_next):
    """
    Based on the existing data, predict the frequency position of the next point:
    - 1 point: constant value prediction
    - 2 points: linear extrapolation
    - 3 points: quadratic extrapolation
    - >= 4 points: use the nearest 4 points for local cubic extrapolation
    """
    k_hist = np.asarray(k_hist, dtype=float)
    omega_hist = np.asarray(omega_hist, dtype=complex)

    nh = len(k_hist)
    if nh == 0:
        raise ValueError("Empty history for mode prediction.")
    if nh == 1:
        return omega_hist[-1]

    use = min(4, nh)
    kh = k_hist[-use:]
    wh = omega_hist[-use:]

    if use == 2:
        coef_r = np.polyfit(kh, np.real(wh), 1)
        coef_i = np.polyfit(kh, np.imag(wh), 1)
        return np.polyval(coef_r, k_next) + 1j * np.polyval(coef_i, k_next)

    if use == 3:
        coef_r = np.polyfit(kh, np.real(wh), 2)
        coef_i = np.polyfit(kh, np.imag(wh), 2)
        return np.polyval(coef_r, k_next) + 1j * np.polyval(coef_i, k_next)

    cs_r = CubicSpline(kh, np.real(wh), bc_type="not-a-knot", extrapolate=True)
    cs_i = CubicSpline(kh, np.imag(wh), bc_type="not-a-knot", extrapolate=True)
    return cs_r(k_next) + 1j * cs_i(k_next)


def track_modes_k_scan(state, kk, theta, rel=0, eig_weight=0.35, pred_weight=0.70):
    """
    method=2 Special waveform tracking:
    1. First, calculate the unsorted eigenvalues/eigenvectors for each k-point.
    2. For each tracked waveform branch, predict the position of the next point using historical points.
    3. Construct the cost matrix using "distance of predicted point + overlap of eigenvectors".
    4. Use the Hungarian algorithm to perform the global optimal matching.
    """
    ns = 4 * state.s + 6
    nk = len(kk)
    nvar = ns

    omega_track = np.zeros((nk, ns), dtype=complex)
    eig_track = np.zeros((nk, nvar, ns), dtype=complex)

    for ik, kval in enumerate(kk):
        kx = kval * np.sin(theta)
        kz = kval * np.cos(theta)

        _, _, w_raw, X_raw = pdrfsolver(
            state,
            kx,
            kz,
            rel=rel,
            return_eigenvectors=True,
            raw_output=True,
        )
        X_raw = normalize_columns(X_raw)

        # The first point should be initially ordered by Re(omega)
        if ik == 0:
            idx0 = np.argsort(np.real(w_raw))[::-1]
            omega_track[0, :] = w_raw[idx0]
            eig_track[0, :, :] = X_raw[:, idx0]
            continue

        prev_w = omega_track[ik - 1, :]
        prev_X = normalize_columns(eig_track[ik - 1, :, :])

        pred_w = np.zeros(ns, dtype=complex)
        for im in range(ns):
            pred_w[im] = predict_next_omega(kk[:ik], omega_track[:ik, im], kk[ik])

        cost = np.zeros((ns, ns), dtype=float)

        for im in range(ns):
            scale_pred = max(1.0, np.abs(pred_w[im]), np.abs(prev_w[im]))
            scale_pred = 1.0
            dw_pred = np.abs(w_raw - pred_w[im]) / scale_pred

            scale_prev = max(1.0, np.abs(prev_w[im]))
            scale_prev = 1.0
            dw_prev = np.abs(w_raw - prev_w[im]) / scale_prev

            overlap = np.abs(prev_X[:, im].conj() @ X_raw)
            overlap = np.clip(overlap, 0.0, 1.0)

            cost[im, :] = (
                pred_weight * dw_pred
                + (1.0 - pred_weight) * dw_prev
                + eig_weight * (1.0 - overlap)
            )

        rows, cols = linear_sum_assignment(cost)

        for row, col in zip(rows, cols):
            omega_track[ik, row] = w_raw[col]

            # Phase alignment is performed to make the eigenvectors more continuous along k, 
            # facilitating the subsequent phase-type polarization diagnosis.
            vec = X_raw[:, col]
            phase_ref = np.vdot(prev_X[:, row], vec)
            if np.abs(phase_ref) > 0:
                vec = vec * np.exp(-1j * np.angle(phase_ref))

            eig_track[ik, :, row] = vec

    return omega_track, eig_track


def compute_mode_diagnostics(state, eig_track):
    """
    The wave mode properties are calculated based on the intrinsic vectors that have been tracked. 
    Here, the default background magnetic field is along the z-axis, therefore: 
    - B_parallel = Bz
    - Ex, Ey represent the horizontal principal components
    """
    # nk, nvar, ns = eig_track.shape
    # ind2 = 4 * state.s
    # eps = 1e-14

    # diagnostics = {
    #     "EyEx": np.full((nk, ns), np.nan, dtype=float),
    #     "EzEx": np.full((nk, ns), np.nan, dtype=float),
    #     "phaseEyEx": np.full((nk, ns), np.nan, dtype=float),
    #     "density_comp": np.full((nk, ns), np.nan, dtype=float),
    #     "magnetic_comp": np.full((nk, ns), np.nan, dtype=float),
    #     "EB_ratio": np.full((nk, ns), np.nan, dtype=float),
    #     "ByBx": np.full((nk, ns), np.nan, dtype=float),
    #     "phaseByBx": np.full((nk, ns), np.nan, dtype=float),
    #     "BzBx": np.full((nk, ns), np.nan, dtype=float),
    # }

    # for ik in range(nk):
    #     for im in range(ns):
    #         vec = eig_track[ik, :, im]
    #         scale = np.max(np.abs(vec))
    #         if scale < eps:
    #             continue
    #         vec = vec / scale

    #         dn = vec[0:ind2:4]
    #         E = vec[ind2:ind2 + 3]
    #         B = vec[ind2 + 3:ind2 + 6]

    #         ex, ey, ez = E
    #         bx, by, bz = B

    #         e_norm = np.linalg.norm(E)
    #         b_norm = np.linalg.norm(B)

    #         dn_rel = dn / np.maximum(state.ns0, eps)

    #         if np.abs(ex) >= eps:
    #             diagnostics["EyEx"][ik, im] = np.abs(ey / ex)
    #             diagnostics["EzEx"][ik, im] = np.abs(ez / ex)
    #             diagnostics["phaseEyEx"][ik, im] = np.degrees(np.angle(ey / ex))

    #         diagnostics["density_comp"][ik, im] = np.linalg.norm(dn_rel) / (b_norm + eps)
    #         diagnostics["magnetic_comp"][ik, im] = np.abs(bz) ** 2 / (b_norm**2 + eps)
    #         diagnostics["EB_ratio"][ik, im] = e_norm / (np.sqrt(state.c2) * b_norm + eps)
    #         diagnostics["ByBx"][ik, im] = np.abs(by / bx) if np.abs(bx) >= eps else np.nan
    #         diagnostics["phaseByBx"][ik, im] = np.degrees(np.angle(by / bx)) if np.abs(bx) >= eps else np.nan
    #         diagnostics["BzBx"][ik, im] = np.abs(bz / bx) if np.abs(bx) >= eps else np.nan

    nk, nvar, ns = eig_track.shape
    ind2 = 4 * state.s
    eps_abs = 1e-14              
    rel_threshold = 1e-6          

    diagnostics = {
        "EyEx":          np.full((nk, ns), np.nan, dtype=float),
        "EzEx":          np.full((nk, ns), np.nan, dtype=float),
        "E_ellipse_angle":     np.full((nk, ns), np.nan, dtype=float),
        "density_comp":  np.full((nk, ns), np.nan, dtype=float),
        "magnetic_comp": np.full((nk, ns), np.nan, dtype=float),
        "EB_ratio":      np.full((nk, ns), np.nan, dtype=float),
        "ByBx":          np.full((nk, ns), np.nan, dtype=float),
        "B_ellipse_angle":     np.full((nk, ns), np.nan, dtype=float),
        "BzBx":          np.full((nk, ns), np.nan, dtype=float),
    }

    for ik in range(nk):
        for im in range(ns):
            vec = eig_track[ik, :, im]
            scale = np.max(np.abs(vec))
            if scale < eps_abs:
                continue
            vec = vec / scale

            dn = vec[0:ind2:4]
            E  = vec[ind2:ind2 + 3]
            B  = vec[ind2 + 3:ind2 + 6]
            ex, ey, ez = E
            bx, by, bz = B

            e_norm = np.linalg.norm(E)
            b_norm = np.linalg.norm(B)
            dn_rel = dn / np.maximum(state.ns0, eps_abs)

            if np.abs(ex) >= rel_threshold * e_norm and e_norm > eps_abs:
                diagnostics["EyEx"][ik, im]      = np.abs(ey / ex)
                diagnostics["EzEx"][ik, im]      = np.abs(ez / ex)
                S1 = np.abs(ex)**2 - np.abs(ey)**2
                S2 = 2.0 * np.real(ex * np.conj(ey))
                psi_raw  = 0.5 * np.degrees(np.arctan2(S2, S1))    # ∈ (-90, 90]
                psi_fold = psi_raw % 180.0                           # ∈ [0, 180)
                if psi_fold > 160.0:
                    psi_fold -= 180.0
                diagnostics["E_ellipse_angle"][ik, im] = psi_fold   

            diagnostics["density_comp"][ik, im]  = np.linalg.norm(dn_rel) / (b_norm + eps_abs)
            diagnostics["magnetic_comp"][ik, im] = np.abs(bz)**2 / (b_norm**2 + eps_abs)
            diagnostics["EB_ratio"][ik, im]      = e_norm / (np.sqrt(state.c2) * b_norm + eps_abs)

            if np.abs(bx) >= rel_threshold * b_norm and b_norm > eps_abs:
                S1 = np.abs(bx)**2 - np.abs(by)**2
                S2 = 2.0 * np.real(bx * np.conj(by))
                psi_raw  = 0.5 * np.degrees(np.arctan2(S2, S1))    # ∈ (-90, 90]
                psi_fold = psi_raw % 180.0                           # ∈ [0, 180)
                if psi_fold > 160.0:
                    psi_fold -= 180.0
                diagnostics["ByBx"][ik, im]      = np.abs(by / bx)
                diagnostics["BzBx"][ik, im]      = np.abs(bz / bx)
                diagnostics["B_ellipse_angle"][ik, im] = psi_fold   
                
    return diagnostics


def plot_method2_results(kk, omega_track, diagnostics, state, theta):
    ns = omega_track.shape[1]
    nk = len(kk)
    styles = build_mode_styles(ns, nk)

    # -------- dispersion relation --------
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 4.8), sharex=True)

    for im in range(ns):
        style = styles[im]
        color = style["color"]
        marker = style["marker"]
        markevery = style["markevery"]

        wr = np.real(omega_track[:, im])
        wi = np.imag(omega_track[:, im])

        # real part
        axes1[0].plot(
            kk, wr,
            color=color,
            linewidth=1.15,
            alpha=0.85,
            zorder=1,
        )
        axes1[0].plot(
            kk, wr,
            linestyle="None",
            marker=marker,
            markevery=markevery,
            markersize=4.2,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.95,
            zorder=2,
        )

        # imaginary part
        axes1[1].plot(
            kk, wi,
            color=color,
            linewidth=1.15,
            alpha=0.85,
            zorder=1,
        )
        axes1[1].plot(
            kk, wi,
            linestyle="None",
            marker=marker,
            markevery=markevery,
            markersize=4.2,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.95,
            zorder=2,
        )

    axes1[0].set_xlabel("kc")
    axes1[0].set_ylabel(r"$\omega_r / \omega_{ce}$")
    axes1[0].set_title(f"(a) tracked $\\omega_r-k$, theta={theta * 180 / np.pi:.1f} deg")
    axes1[0].grid(True, alpha=0.25)

    axes1[1].set_xlabel("kc")
    axes1[1].set_ylabel(r"$\omega_i / \omega_{ce}$")
    if state.s > 1:
        axes1[1].set_title(f"(b) tracked $\\omega_i-k$, mi/me={state.ms[1] / state.ms[0]:.3g}")
    else:
        axes1[1].set_title("(b) tracked $\\omega_i-k$")
    axes1[1].grid(True, alpha=0.25)

    # mode number labels on the right side of the dispersion curves
    add_mode_labels(axes1[0], kk[-1], np.real(omega_track[-1, :]), styles, prefix="m")
    add_mode_labels(axes1[1], kk[-1], np.imag(omega_track[-1, :]), styles, prefix="m")

    fig1.tight_layout(rect=[0, 0, 0.90, 1])

    # -------- Wave mode property diagram: forward propagation branches only --------
    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 8), sharex=True)
    axes2 = axes2.ravel()

    items = [
        ("EyEx", r"$|E_y/E_x|$", True),
        ("EzEx", r"$|E_z/E_x|$", True),
        ("E_ellipse_angle", r"$\psi$ [deg]", False),
        ("density_comp", r"$||\delta n/n_0|| \,/\, ||\delta B||$", True),
        ("magnetic_comp", r"$|\delta B_{\parallel}|^2 / |\delta B|^2$", False),
        ("EB_ratio", r"$|\delta E| / (c |\delta B|)$", True),
        ("ByBx", r"$|B_y/B_x|$", True),
        ("B_ellipse_angle", r"$\psi$ [deg]", False),
        ("BzBx", r"$|B_z/B_x|$", True),
    ]

    forward_tol = 1e-8
    forward_mask = np.real(omega_track) >= -forward_tol

    for ax, (key, ylabel, use_log) in zip(axes2, items):
        for im in range(ns):
            y = diagnostics[key][:, im].copy()
            y[~forward_mask[:, im]] = np.nan

            if use_log:
                y[y <= 0] = np.nan

            ax.plot(
                kk, y,
                color=styles[im]["color"],
                linewidth=1.2,
                alpha=0.9,
            )

            ax.plot(
                kk, y,
                linestyle="None",
                marker=styles[im]["marker"],
                markevery=styles[im]["markevery"],
                markersize=3.6,
                markerfacecolor="white",
                markeredgecolor=styles[im]["color"],
                markeredgewidth=0.8,
            )
            if key in ("E_ellipse_angle", "B_ellipse_angle"):
                ax.set_title(f"angle between major axes of $\mathbf{{{key[0].upper()}}}$ ellipse and x-axis")

        ax.set_xlabel("kc")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

        if use_log:
            ax.set_yscale("log")

    axes2[2].set_ylim(-180.0, 180.0)
    axes2[4].set_ylim(0.0, 1.05)

    fig2.suptitle("Mode diagnostics for forward-propagating branches (Re(omega) >= 0)", y=0.98)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])

    plt.show()


def pdrf(method=2, rel=0, input_file="pdrf.in", kmin=kmin, kmax=kmax, dk=dk, theta=theta):
    """
    Main entrance, corresponding to MATLAB's pdrf 
    method:
    1 - Single point k 
    2 - w(k)
    3 - w(theta)
    4 - w(kx, kz)
    """
    state = initial(input_file)

    if method == 2:
        # kmin, kmax, dk = 0.0, 10.0, 0.1
        # theta = np.pi / 3
        kk = np.arange(kmin, kmax + dk, dk)

        omega_track, eig_track = track_modes_k_scan(state, kk, theta, rel=rel)
        diagnostics = compute_mode_diagnostics(state, eig_track)

        plot_method2_results(kk, omega_track, diagnostics, state, theta)

        return kk, omega_track, diagnostics

    elif method == 3:
        k = 2.0
        dthe = np.pi / 100
        tt = np.arange(0, np.pi / 2 + dthe, dthe)
        nt = len(tt)

        ww1 = np.zeros((nt, 4 * state.s + 6), dtype=complex)
        ww2 = np.zeros((nt, 4 * state.s + 6), dtype=complex)

        for it, theta in enumerate(tt):
            kx = k * np.sin(theta)
            kz = k * np.cos(theta)
            _, _, w1, w2 = pdrfsolver(state, kx, kz, rel)
            ww1[it, :] = w1
            ww2[it, :] = w2

        ns = 2 * state.s + 1

        plt.figure(figsize=(12, 4))
        cmap = plt.get_cmap("tab20", ns)

        for is_ in range(ns):
            color = cmap(is_)
            plt.subplot(1, 2, 1)
            plt.plot(tt, np.real(ww1[:, is_]), color=color, linewidth=2)
            plt.xlabel("theta")
            plt.ylabel("omega_r / omega_ce")
            plt.title(f"(c) omega_r-theta, kc={k}")
            plt.xlim(tt.min(), tt.max())

            plt.subplot(1, 2, 2)
            plt.plot(tt, np.imag(ww2[:, is_]), ".", color=color)
            plt.xlabel("theta")
            plt.ylabel("omega_i / omega_ce")
            plt.title("(d) omega_i-theta")
            plt.xlim(tt.min(), tt.max())

        plt.tight_layout()
        plt.show()
        return ww1, ww2

    elif method == 4:
        kkx, kkz = np.meshgrid(
            np.arange(0.01, 3.0 + 0.04, 0.04),
            np.arange(0.01, 2.0 + 0.04, 0.04)
        )
        row, col = kkx.shape

        ww1 = np.zeros((row, col, 4 * state.s + 6), dtype=complex)
        ww2 = np.zeros((row, col, 4 * state.s + 6), dtype=complex)

        for ir in range(row):
            for ic in range(col):
                kx = kkx[ir, ic]
                kz = kkz[ir, ic]
                _, _, w1, w2 = pdrfsolver(state, kx, kz, rel)
                ww1[ir, ic, :] = w1
                ww2[ir, ic, :] = w2

        ns = 4 * state.s + 6

        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        for is_ in range(ns):
            ax1.plot_surface(kkx, kkz, np.real(ww1[:, :, is_]), alpha=0.5)
            ax2.plot_surface(kkx, kkz, np.imag(ww2[:, :, is_]), alpha=0.5)

        ax1.set_xlabel("k_x c")
        ax1.set_ylabel("k_z c")
        ax1.set_zlabel("omega_r / omega_ce")
        ax1.set_title("Re(omega)")

        ax2.set_xlabel("k_x c")
        ax2.set_ylabel("k_z c")
        ax2.set_zlabel("omega_i / omega_ce")
        ax2.set_title("Im(omega)")

        plt.tight_layout()
        plt.show()
        return kkx, kkz, ww1, ww2

    else:
        k = 0.1
        theta = np.pi / 3
        kx = k * np.sin(theta)
        kz = k * np.cos(theta)

        M, A, w1, w2 = pdrfsolver(state, kx, kz, rel)
        return M, A, w1, w2


if __name__ == "__main__":
    result = pdrf(method=2, rel=0, input_file="pdrf.in", kmin=kmin, kmax=kmax, dk=dk, theta=theta)
