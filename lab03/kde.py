import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV

################################## TASK 1 ##################################
# a) data generation
# idea: z p-stwem 0.9 losujemy z N(5, 1) i z p-stwem 0.1 losujemy z N(10, 1)

n = 200
cluster = np.random.binomial(1, 0.1, size=n)   # losujemy 200 zmiennych Bernoulliego Zi ~ Bern(0.1) -> czyli: Zi = 1 z p-stwem 0.1 i Zi = 0 z p-stwem 0.9
x = np.zeros(n)

for i in np.arange(n):
    if cluster[i] == 0:
        x[i] = np.random.normal(loc=5, scale=1, size=1)[0]   # dla 0 czyli p-stwo 0.9 (wyzej) robimy rozklad N(5, 1)
    else:
        x[i] = np.random.normal(loc=10, scale=1, size=1)[0]

plt.hist(x, bins=30, density=True, alpha=0.7, color="grey")
plt.title("Sample from mixture: 0.9*N(5,1) + 0.1*N(10,1)")
plt.show()

# # b) kde
x = x.reshape(-1, 1)
kde = KernelDensity(kernel="gaussian", bandwidth=0.5)
kde.fit(x)

# testing points for evaluation
x0 = np.linspace(0, 15, 500).reshape(-1, 1)

# the function returns log densities - we go back to og
log_density = kde.score_samples(x0)
kde_density = np.exp(log_density)

# true density
true_density = 0.9 * norm.pdf(x0.flatten(), 5, 1) + 0.1 * norm.pdf(x0.flatten(), 10, 1)

plt.figure(figsize=(8, 5))

plt.plot(x0, true_density, label="True density", linewidth=3)
plt.plot(x0, kde_density, label="KDE estimate", linewidth=2)
plt.hist(x, bins=30, density=True, alpha=0.3, label="Histogram", color="grey")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Density estimation")
plt.legend()
# plt.show()

# ##
# # c) mse
def mse(n=100, K=200, kernel_type="gaussian", bandwidth=0.5, x=None):
    if x is None:
        # data generation - from a)
        cluster = np.random.binomial(1, 0.1, size=n)
        x = np.zeros(n)

        for i in np.arange(n):
            if cluster[i] == 0:
                x[i] = np.random.normal(loc=5, scale=1, size=1)[0]
            else:
                x[i] = np.random.normal(loc=10, scale=1, size=1)[0]
            
    # testing points
    x_test = np.random.uniform(low=2, high=12, size=K)

    # true density
    true_density = 0.9 * norm.pdf(x_test.flatten(), 5, 1) + 0.1 * norm.pdf(x_test.flatten(), 10, 1)

    x = x.reshape(-1, 1)
    kde = KernelDensity(kernel=kernel_type, bandwidth=bandwidth)
    kde.fit(x)

    x_test = x_test.reshape(-1, 1)
    log_kde_density = kde.score_samples(x_test)
    kde_density = np.exp(log_kde_density)

    # error
    mse = (1 / K) * ((true_density - kde_density)**2).sum()

    return mse

# # d) how the error depends on the sample size n 
n_val = [10, 50, 100, 500, 1000]
def kde_diff_n(n_val=n_val):
    mse_val = []
    for n in n_val:
        mse_val.append(mse(n=n))

    plt.plot(n_val, mse_val)
    plt.xlabel("n value - given sample size")
    plt.ylabel("MSE")
    plt.title("MSE value change across different sample size")
    plt.show()

# e)
kernels = ["gaussian", "tophat", "epanechnikov"]
def kde_diff_kernels(x, kernels=kernels):
    mse_val = []
    for kernel_type in kernels:
        mse_val.append(mse(kernel_type=kernel_type, x=x))

    plt.plot(kernels, mse_val, marker="o")
    plt.xlabel("Kernel type")
    plt.ylabel("MSE")
    plt.title("MSE value change across different kernel types")
    plt.show()

bandwidth_val = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2]
def kde_diff_bandwidths(x, bandwidth_val=bandwidth_val):
    mse_val = []
    for bandwidth in bandwidth_val:
            mse_val.append(mse(bandwidth=bandwidth, x=x))

    plt.plot(bandwidth_val, mse_val)
    plt.xlabel("Bandwidth (h)")
    plt.ylabel("MSE")
    plt.title("MSE value change across different bandwidths")
    plt.show()

# bandwidth - cross validation
def estimate_bandwidth(x):
    x = x.reshape(-1, 1)

    params = {
        "bandwidth": np.linspace(0.05, 1.5, 30),
    } 

    grid = GridSearchCV(
        estimator=KernelDensity(kernel="gaussian"),
        param_grid=params,
        cv=5,
    )

    grid.fit(x)

    print("Best bandwidth: ", grid.best_params_["bandwidth"])
    return grid.best_params_["bandwidth"]

################################## TASK 2 ##################################
def sample1(n=200):
    # generate data
    cluster = np.random.binomial(1, 0.1, size=n)
    x = np.zeros(n)

    for i in np.arange(n):
        if cluster[i] == 0:
            x[i] = np.random.normal(loc=5, scale=1, size=1)[0]
        else:
            x[i] = np.random.normal(loc=10, scale=1, size=1)

    return x

def sample2(x, n=200, k=1000, bandwidth=0.5):
    x_prim = np.zeros(k)

    for j in range(k):
        i = np.random.randint(low=0, high=n)
        epsilon = np.random.normal(loc=0, scale=1, size=1)[0]
        x_j = x[i] + epsilon * bandwidth
        x_prim[j] = x_j

    return x_prim

def kde_fitting_mse(x, bandwidth=0.5):
    # fit kde
    x = x.reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(x)

    # testing points
    K = 500
    x_test = np.linspace(0, 15, K).reshape(-1, 1)

    # the function returns log densities - we go back to og
    log_density = kde.score_samples(x_test)
    kde_density = np.exp(log_density)

    # true density
    true_density = 0.9 * norm.pdf(x0.flatten(), 5, 1) + 0.1 * norm.pdf(x0.flatten(), 10, 1)

    mse = (1 / K) * ((true_density - kde_density)**2).sum()

    return mse


    
