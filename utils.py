# python
import matplotlib.pyplot as plt
import numpy as np

#for comparison of two samples
def energy_distance(X, Y):
    """
    X: array (n, d)
    Y: array (m, d)
    returns: energy distance (scalar)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    n, m = X.shape[0], Y.shape[0]

    # pairwise distances
    d_xy = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2)  # (n, m)
    d_xx = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)  # (n, n)
    d_yy = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)  # (m, m)

    # unbiased estimates (ignore diagonal for xx, yy)
    term_xy = 2.0 * d_xy.mean()
    if n > 1:
        term_xx = d_xx[np.triu_indices(n, k=1)].mean()
    else:
        term_xx = 0.0
    if m > 1:
        term_yy = d_yy[np.triu_indices(m, k=1)].mean()
    else:
        term_yy = 0.0

    ed2 = term_xy - term_xx - term_yy  # squared energy distance
    return abs(ed2)


def energy_distance_perm_test(X, Y, n_perm=500, random_state=None):
    """
    Returns (ED, p_value) using a permutation test.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    Y = np.asarray(Y)
    n, m = X.shape[0], Y.shape[0]

    ed_obs = energy_distance(X, Y)

    Z = np.vstack([X, Y])
    N = n + m
    count = 0

    for _ in range(n_perm):
        perm = rng.permutation(N)
        Xp = Z[perm[:n]]
        Yp = Z[perm[n:]]
        ed_perm = energy_distance(Xp, Yp)
        if ed_perm >= ed_obs:
            count += 1

    p_value = (count + 1) / (n_perm + 1)  # permutation p-value
    return ed_obs, p_value


#for plotting samples when d=2 or 3
def plot_3d_samples(X_train, samples, title="3D Samples with all pairwise projections"):
    """
    Plot 3D scatter of (x0,x1,x2) and all three 2D projections:
    (x0,x1), (x0,x2), (x1,x2).
    """
    if X_train.shape[1] != 3 or samples.shape[1] != 3:
        raise ValueError("Both X_train and samples must have shape (N, 3) for dim=3.")

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(title, y=0.98, fontsize=12)

    # 3D scatter (top-left)
    ax3d = fig.add_subplot(221, projection="3d")
    ax3d.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2],
                 s=8, c="steelblue", alpha=0.35, label="X_train")
    ax3d.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
                 s=20, c="crimson", alpha=0.7, label="samples")
    ax3d.set_xlabel("x0")
    ax3d.set_ylabel("x1")
    ax3d.set_zlabel("x2")
    ax3d.set_title("3D Scatter")
    ax3d.legend(loc="upper left")

    # Pairwise projections
    pairs = [
        (0, 1, 222, "(x0, x1)"),
        (0, 2, 223, "(x0, x2)"),
        (1, 2, 224, "(x1, x2)"),
    ]
    for i, j, code, subtitle in pairs:
        ax = fig.add_subplot(code)
        ax.scatter(X_train[:, i], X_train[:, j], s=10, c="steelblue", alpha=0.35, label="X_train")
        ax.scatter(samples[:, i], samples[:, j], s=20, c="crimson", alpha=0.7, label="samples")
        ax.set_xlabel(f"x{i}")
        ax.set_ylabel(f"x{j}")
        ax.set_title(f"Projection {subtitle}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_2d_samples(X_train, samples, title: str = "2D Samples vs X_train"):
    """
    Overlay scatter plots for X_train and generated samples when dim == 2.
    """
    if X_train.shape[1] != 2 or samples.shape[1] != 2:
        raise ValueError("Both X_train and samples must have shape (N, 2) for dim=2.")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_train[:, 0], X_train[:, 1], s=10, c="steelblue", alpha=0.35, label="X_train")
    ax.scatter(samples[:, 0], samples[:, 1], s=20, c="crimson", alpha=0.7, label="samples")

    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.2)

    plt.tight_layout()
    plt.show()




