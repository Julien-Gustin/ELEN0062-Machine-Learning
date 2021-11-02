from scipy.stats import norm, uniform
import numpy as np
import matplotlib.pyplot as plt

def f(x, seed=42):
    e = norm.rvs(loc=0, scale=0.5, size=len(x), random_state=seed)
    return np.sin(2*x) + x*np.cos(x - 1) + e

def make_data(N_samples, seed=42):
    X = uniform.rvs(loc=-10, scale=20, size=N_samples, random_state=seed)
    X = np.round(X, decimals=1)
    Y = f(X, seed=seed)
    return X, Y

def plot_data(X, Y):
    plt.figure(figsize=(17, 10), dpi=80)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X, Y, alpha=0.5, s=5)



