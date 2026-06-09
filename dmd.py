import numpy as np
from pydmd import DMD as _DMD


def dmd(states, r=0, dt=1.0):
    """
    DMD on particle simulation snapshots, backed by PyDMD.

    Parameters
    ----------
    states : list of ndarray shape (n, 3), length T+1
    r      : SVD rank — 0 = optimal hard threshold (Gavish-Donoho),
             positive int = fixed rank, -1 = no truncation
    dt     : timestep between snapshots

    Returns
    -------
    dict:
        model          – the fitted PyDMD object (swap for HODMD, OptDMD, etc.)
        modes          (3n, r) complex  – DMD spatial modes
        eigenvalues    (r,)    complex  – discrete-time eigenvalues λ
        frequencies    (r,)    float    – oscillation frequency (1/time units)
        growth_rates   (r,)    float    – continuous growth (+) / decay (-) rate
        amplitudes     (r,)    complex  – mode amplitudes fitted to x₀
        singular_values (k,)   float    – singular values of X1 (for rank plot)
        reconstruction (T+1, 3n) complex – full reconstructed trajectory
    """
    X = np.array([s.flatten() for s in states]).T   # (3n, T+1)

    model = _DMD(svd_rank=r)
    model.fit(X)

    # Singular values for diagnostics (PyDMD doesn't expose them directly)
    _, sigma, _ = np.linalg.svd(X[:, :-1], full_matrices=False)

    # Continuous-time eigenvalues
    omega = np.log(model.eigs) / dt

    # Reconstruct all T+1 snapshots via Vandermonde (PyDMD's reconstructed_data
    # covers only the training window, so we build it manually to match states)
    T = len(states)
    vander = np.array([model.eigs**t for t in range(T)])           # (T, r)
    reconstruction = (model.modes @ (model.amplitudes * vander).T).T  # (T, 3n)

    return {
        'model': model,
        'modes': model.modes,
        'eigenvalues': model.eigs,
        'frequencies': omega.imag / (2 * np.pi),
        'growth_rates': omega.real,
        'amplitudes': model.amplitudes,
        'singular_values': sigma,
        'reconstruction': reconstruction,
    }
