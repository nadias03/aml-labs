import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def y_generation(X, n=50):
    y = np.zeros(n)
    beta0 = 0.5
    beta = np.array([1, 1, 1, 1, 1])
    for i in range(n):
        s = beta0 + beta @ X[i, :]   # i-ty wiersz i wszystkie kolumny
        p = 1 / (1 + np.exp(-s))
        y[i] = np.random.binomial(n=1, p=p, size=1)

    return y

def experiment(L=100, n=50):
    mse = np.zeros(L)
    for i in range(L):
        X = np.random.normal(loc=0, scale=1, size=(n, 5))
        y = y_generation(X=X, n=n)

        logistic_regression = LogisticRegression(penalty=None)
        logistic_regression.fit(X, y)

        beta_true = np.array([0.5, 1, 1, 1, 1, 1])
        beta_hat = np.concatenate(([logistic_regression.intercept_[0]], logistic_regression.coef_[0]))

        mse[i] = np.linalg.norm(beta_hat - beta_true) ** 2

    return mse.mean()

def mse_depending_on_n(n_vals):
    n_vals = np.array(n_vals)
    mse_arr = np.zeros(n_vals.shape[0])
    for i in range(n_vals.shape[0]):
        mse_arr[i] = experiment(L=100, n=n_vals[i])    

    plt.plot(n_vals, mse_arr)
    plt.xlabel("n value - sample size")
    plt.ylabel("MSE")
    plt.title("MSE value change across different sample size")
    plt.show()

def compare_experiments(n_vals, L=100):
    n_vals = np.array(n_vals)
    mse_normal = np.zeros(n_vals.shape[0])
    mse_3vars = np.zeros(n_vals.shape[0])
    
    for j in range(n_vals.shape[0]):
        n = n_vals[j]
        mse_normal_exp = np.zeros(L)
        mse_3vars_exp = np.zeros(L)

        for i in range(L):
            X = np.random.normal(loc=0, scale=1, size=(n, 5))
            y = y_generation(X=X, n=n)

            logistic_regression_normal = LogisticRegression(penalty=None)
            logistic_regression_normal.fit(X, y)
            beta_true_normal = np.array([0.5, 1, 1, 1, 1, 1])
            beta_hat_normal = np.concatenate((
                [logistic_regression_normal.intercept_[0]],
                logistic_regression_normal.coef_[0],
            ))
            mse_normal_exp[i] = np.linalg.norm(beta_hat_normal - beta_true_normal) ** 2

            X_reduced = X[:, :3]
            logistic_regression_3vars = LogisticRegression(penalty=None)
            logistic_regression_3vars.fit(X_reduced, y)
            beta_true_3vars = np.array([0.5, 1, 1, 1])
            beta_hat_3vars = np.concatenate((
                [logistic_regression_3vars.intercept_[0]],
                logistic_regression_3vars.coef_[0],
            ))

            mse_3vars_exp[i] = np.linalg.norm(beta_hat_3vars - beta_true_3vars) ** 2

        mse_normal[j] = mse_normal_exp.mean()
        mse_3vars[j] = mse_3vars_exp.mean()

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(n_vals, mse_normal)
    axes[0].set_xlabel("n value - sample size")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Full model (5 variables)")

    axes[1].plot(n_vals, mse_3vars)
    axes[1].set_xlabel("n value - sample size")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Reduced model (3 variables)")

    plt.show()