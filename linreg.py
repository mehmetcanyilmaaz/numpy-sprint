# %% data
import numpy as np
import matplotlib.pyplot as plt

def make_data(beta_true, n=200, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    d = len(beta_true) - 1                      # features, excluding intercept
    features = rng.standard_normal((n, d))      # (n, d)
    X = np.column_stack([np.ones(n), features]) # (n, d+1): ones column first
    y = X @ beta_true + noise * rng.standard_normal(n)  # (n,)
    return X, y


beta_true = np.array([3.0, 2.0, -1.0, 0.5])
X, y = make_data(beta_true)
print("X shape:", X.shape, "y shape:", y.shape)
print("first column all ones:", np.all(X[:, 0] == 1))

# %% normal equations
def normal_equations(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

# Test on the 3-point example you solved by hand
X3 = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
y3 = np.array([1.0, 2.0, 4.0])
print("3-point:", normal_equations(X3, y3), "expected [0.8333, 1.5]")

beta_ne = normal_equations(X, y)
print("normal eq:", beta_ne)
print("true:     ", beta_true)

residual = y - X @ beta_ne
print("X^T residual (should be ~0):", X.T @ residual)
print("lstsq agrees:", np.allclose(beta_ne, np.linalg.lstsq(X, y, rcond=None)[0]))

# %% gradient descent
def gradient_descent(X, y, lr=0.1, max_steps=10_000, tol=1e-8):
    n, d = X.shape
    beta = np.zeros(d)
    losses = []
    for step in range(max_steps):
        residual = X @ beta - y                 # (n,)
        losses.append(np.mean(residual ** 2))
        if not np.isfinite(losses[-1]):         # diverged, stop
            break
        grad = (2 / n) * X.T @ residual         # (d,)
        if np.linalg.norm(grad) < tol:          # converged, stop
            break
        beta = beta - lr * grad
    return beta, losses


# Hand-trace check: one step on the 3-point example
b1, _ = gradient_descent(X3, y3, lr=0.1, max_steps=1)
print("one step:", b1, "expected [0.467, 0.667]")

beta_gd, losses = gradient_descent(X, y, lr=0.1)
print("GD:", beta_gd, "in", len(losses), "steps")

# %% comparing and predicting the cliff
print("GD matches normal eq:", np.allclose(beta_gd, beta_ne, atol=1e-6))

H = (2 / len(y)) * X.T @ X
eigs = np.linalg.eigvalsh(H)
lr_max = 2 / eigs.max()
print("Hessian eigenvalues:", eigs)
print("predicted max lr:", lr_max) # cliff is this max step size, above which GD diverges

_, losses_ok = gradient_descent(X, y, lr=0.95 * lr_max, max_steps=200)
_, losses_bad = gradient_descent(X, y, lr=1.05 * lr_max, max_steps=200)
print("just below cliff, final loss:", losses_ok[-1])
print("just above cliff, final loss:", losses_bad[-1])

# %% plotting

L_star = np.mean((X @ beta_ne - y) ** 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(losses)
ax1.set(title="Loss (lr=0.1)", xlabel="step", ylabel="MSE")

gap = np.maximum(np.array(losses) - L_star, 1e-16)
ax2.semilogy(gap, label="lr=0.1")
ax2.semilogy(np.maximum(np.array(losses_bad) - L_star, 1e-16), label="above cliff")
ax2.set(title="Loss minus optimum (log scale)", xlabel="step")
ax2.legend()

plt.tight_layout()
plt.savefig("loss_curves.png")
plt.show()

# %%
