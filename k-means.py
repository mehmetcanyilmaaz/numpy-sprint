# %% data
import numpy as np
import matplotlib.pyplot as plt


def make_blobs(centers, n_per=100, spread=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for j, c in enumerate(centers):
        X_parts.append(c + spread * rng.standard_normal((n_per, len(c))))  # (n_per, d)
        y_parts.append(np.full(n_per, j))                                   # true label j
    return np.vstack(X_parts), np.concatenate(y_parts)


true_centers = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
X, y_true = make_blobs(true_centers)
print("X:", X.shape, "y_true:", y_true.shape)

# %% assinging
def assign(X, C):
    diff = X[:, None, :] - C[None, :, :]   # (n, 1, d) - (1, k, d) -> (n, k, d)
    sq_dist = (diff ** 2).sum(axis=2)      # (n, k): squared distance point i -> centroid j
    return sq_dist.argmin(axis=1), sq_dist # labels (n,), distances (n, k)


# Hand example: points 1, 2, 9, 10, 11 with centroids 1, 2
X1 = np.array([[1.0], [2.0], [9.0], [10.0], [11.0]])  # (5, 1): 1-D data, still 2-D array
C1 = np.array([[1.0], [2.0]])                         # (2, 1)
labels1, _ = assign(X1, C1)
print("labels:", labels1, "expected [0 1 1 1 1]")

# %% update centroids

def update(X, labels, C_old):
    C = C_old.copy()
    for j in range(len(C_old)):
        members = X[labels == j]           # boolean mask: rows in cluster j
        if len(members) > 0:               # empty cluster: keep old centroid
            C[j] = members.mean(axis=0)
    return C


print("update:", update(X1, labels1, C1).ravel(), "expected [1. 8.]")

# %% kmeans
def kmeans(X, k, max_iters=100, seed=0, C_init=None):
    if C_init is None:
        rng = np.random.default_rng(seed)
        C = X[rng.choice(len(X), size=k, replace=False)]  # k distinct data points
    else:
        C = C_init.copy()
    labels = None
    inertias = []
    for it in range(max_iters):
        new_labels, sq_dist = assign(X, C)
        inertias.append(sq_dist[np.arange(len(X)), new_labels].sum())  # each point's own distance
        if labels is not None and np.array_equal(new_labels, labels):
            break                                           # assignments stopped changing
        labels = new_labels
        C = update(X, labels, C)
    return C, labels, inertias


C_hand, _, inert_hand = kmeans(X1, k=2, C_init=C1)
print("hand run:", C_hand.ravel(), "expected [1.5 10.]")
print("inertias:", inert_hand, "expected [194, 15, 2.5]")

# %% check
C, labels, inertias = kmeans(X, k=3)
print("rounds:", len(inertias))
print("inertia never increases:", np.all(np.diff(inertias) <= 1e-9))

# Cluster numbers are arbitrary, so match each found centroid to its nearest true center
match, d = assign(C, true_centers)          # reuse assign: (k, k) distances
for j in range(3):
    print(f"found {C[j].round(2)} -> true {true_centers[match[j]]}, off by {np.sqrt(d[j, match[j]]):.2f}")
print("point accuracy:", np.mean(match[labels] == y_true))

# %% plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap="viridis")
ax1.scatter(C[:, 0], C[:, 1], marker="x", s=200, c="red", label="found")
ax1.scatter(true_centers[:, 0], true_centers[:, 1], marker="+", s=200, c="black", label="true")
ax1.set(title="k-means clusters", aspect="equal")
ax1.legend()

ax2.plot(inertias, marker="o")
ax2.set(title="Inertia per round", xlabel="round", ylabel="sum of squared distances")

plt.tight_layout()
plt.savefig("kmeans.png")
plt.show()

# %% bad local minima
close = np.array([[0.0, 0.0], [2.5, 0.0], [8.0, 0.0]])
Xc, _ = make_blobs(close, seed=1)
for s in range(10):
    _, _, inert = kmeans(Xc, k=3, seed=s)
    print(f"seed {s}: final inertia {inert[-1]:.1f}")

