import numpy as np


def generate_syn_data(n=100, s=1, d1=10, d2=20, p=3, reproduce=0 ):
    if reproduce == 1:
        np.random.seed(66)

    X1 = np.random.randn(n, d1)
    norm_X1 = np.linalg.norm(X1, axis=1)
    # Generate Y (one-hot encoding)

    a = 2.5
    b = 3.5
    Y0 = np.zeros(n)
    Y0[norm_X1 <= a] = 0
    Y0[(norm_X1 >= a) & (norm_X1 <= b)] = 1
    Y0[norm_X1 >= b] = 2

    Y_test = Y0
    Y_one_hot = np.eye(p)[Y0.astype(int)]

    # Generate C(X1) and X2
    A = np.random.uniform(-2, 2, (d2, p))
    B = np.random.uniform(-2, 2, (d2, s))
    N = np.random.randn(n, d2)
    # N = np.zeros([n, d2],dtype=float)
    X2 = np.zeros((n, d2))

    if s == 0:
        for i in range(n):
            C_X1 = A
            X2[i, :] = C_X1 @ Y_one_hot[i, :] + N[i, :]
    else:
        for i in range(n):
            j_vals = np.arange(s).reshape(s, 1)
            k_vals = np.arange(p).reshape(1, p)
            g_X1 = max(X1[i, :]) * np.sin(2 * np.pi * (j_vals / s * np.min(X1[i, :]) + 2 * np.pi * k_vals / p))
            # Calculate C(X1) correctly
            C_X1 = A + B @ g_X1  # Resulting in a (d2, p) matrix
            X2[i, :] = C_X1 @ Y_one_hot[i, :] + N[i, :]

    return X1, X2, Y_one_hot, Y_test