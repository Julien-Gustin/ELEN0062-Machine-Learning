from scipy.stats import norm, uniform
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def f(x, scale=0.5, seed=42):
    e = norm.rvs(loc=0, scale=scale, size=len(x), random_state=seed)
    return np.sin(2*x) + x*np.cos(x - 1) + e

def make_data(N_samples, scale=0.5, seed=42):
    """Data generator

    Args:
        N_samples: 
        scale: Standard deviation Defaults to 0.5.
        seed: Random seed Defaults to 42.

    Returns:
        X, Y
    """
    X = uniform.rvs(loc=-10, scale=20, size=N_samples, random_state=seed)
    Y = f(X, scale=scale, seed=seed)
    return X, Y

def plot_data(X, Y):
    plt.figure(figsize=(20, 12), dpi=80)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X, Y, alpha=0.5, s=5)




