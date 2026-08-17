import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from scipy.stats import chi2

# 1. Generate Synthetic Data
np.random.seed(42)
N_samples = 100
mu_true = 5.0
sigma_true = 2.0

data = np.random.normal(loc=mu_true, scale=sigma_true, size=N_samples)


# 2. Define Negative Log-Likelihood (NLL)
def nll(params):
    mu, sigma = params
    if sigma <= 0.1:
        return 1e10
    return -np.sum(-np.log(sigma) - 0.5 * ((data - mu) / sigma) ** 2)

# 3. Global Unconstrained Fit (MLE)
res_global = minimize(nll, [1.0, 1.0], method='Nelder-Mead')
mu_hat, sigma_hat = res_global.x
nll_min = res_global.fun

# 4. Grid Setup for Profiling
n_frames = 80
mu_grid = np.linspace(mu_hat - 0.8, mu_hat + 0.8, n_frames)

profile_sigmas = []
q_mu_vals = []

# Pre-calculate profile values for smooth plotting
for m in mu_grid:
    res = minimize(lambda s: nll([m, s[0]]), [sigma_hat], method='Nelder-Mead')
    sig_cond = res.x[0]
    profile_sigmas.append(sig_cond)
    q_val = 2 * (res.fun - nll_min)
    q_mu_vals.append(q_val)

profile_sigmas = np.array(profile_sigmas)
q_mu_vals = np.array(q_mu_vals)

# 5. Setup Animated Figure (2 Subplots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=100)

# Left Plot: 2D NLL Contour Surface
sigma_grid = np.linspace(sigma_hat - 0.5, sigma_hat + 0.5, 100)
MU, SIGMA = np.meshgrid(np.linspace(mu_hat - 0.8, mu_hat + 0.8, 100), sigma_grid)
Z = np.zeros_like(MU)

for i in range(MU.shape[0]):
    for j in range(MU.shape[1]):
        Z[i, j] = 2 * (nll([MU[i, j], SIGMA[i, j]]) - nll_min)

contours = ax1.contourf(MU, SIGMA, Z, levels=np.linspace(0, 10, 15), cmap='Blues_r', alpha=0.85)
fig.colorbar(contours, ax=ax1, label=r'$-2\Delta\ln L$')

ax1.plot(mu_hat, sigma_hat, 'r*', markersize=12, label=r'Global MLE $(\hat{\mu}, \hat{\sigma})$')
path_line, = ax1.plot([], [], 'r--', lw=2, label=r'Profile Path $\hat{\hat{\sigma}}_\mu$')
current_point, = ax1.plot([], [], 'ro', markersize=8)

ax1.set_xlabel(r'Parameter of Interest $\mu$')
ax1.set_ylabel(r'Nuisance Parameter $\sigma$')
ax1.set_title(r'2D Likelihood Surface & Profiling Path')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.4)

# Right Plot: 1D Profile Likelihood Curve
q_line, = ax2.plot([], [], color='#1f77b4', lw=2.5, label=r'Profile $-2\ln\Lambda(\mu)$')
scan_point, = ax2.plot([], [], 'ro', markersize=8)

cutoff_1s = chi2.ppf(0.6827, df=1)
cutoff_2s = chi2.ppf(0.9545, df=1)

ax2.axhline(cutoff_1s, color='goldenrod', linestyle='--', label=r'68.3% CL ($\Delta = 1.0$)')
ax2.axhline(cutoff_2s, color='crimson', linestyle=':', label=r'95.5% CL ($\Delta = 3.84$)')
ax2.axvline(mu_hat, color='black', linestyle='-', alpha=0.5, label=r'Best Fit $\hat{\mu}$')

ax2.set_xlim(mu_grid[0], mu_grid[-1])
ax2.set_ylim(0, 6)
ax2.set_xlabel(r'Parameter of Interest $\mu$')
ax2.set_ylabel(r'Test Statistic $q(\mu)$')
ax2.set_title(r'Constructed Profile Likelihood Ratio')
ax2.legend(loc='upper center')
ax2.grid(True, linestyle='--', alpha=0.4)


# Animation Update Function
def update(frame):
    # Update profile trajectory on 2D contour
    path_line.set_data(mu_grid[:frame], profile_sigmas[:frame])
    current_point.set_data([mu_grid[frame]], [profile_sigmas[frame]])

    # Update 1D Profile Likelihood Curve
    q_line.set_data(mu_grid[:frame], q_mu_vals[:frame])
    scan_point.set_data([mu_grid[frame]], [q_mu_vals[frame]])

    return path_line, current_point, q_line, scan_point


anim = FuncAnimation(fig, update, frames=len(mu_grid), interval=50, blit=True)
# Create Animation
anim = FuncAnimation(fig, update, frames=len(mu_grid), interval=50, blit=True)

plt.tight_layout()

# --- SAVE TO VIDEO ---
# Option A: Save as MP4 video (20 fps = ~4 second video)
anim.save('/Users/binishbatool/Downloads/profile_likelihood_anim.mp4', writer='ffmpeg', fps=20, dpi=200)

# Option B: Save as GIF
# anim.save("profile_likelihood.gif", writer='pillow', fps=20, dpi=150)

plt.show()

