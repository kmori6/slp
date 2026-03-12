import numpy as np
from numba import njit


@njit(cache=True)
def monotonic_alignment_search(value: np.ndarray) -> np.ndarray:
    """Monotonic alignment search described in J. Kim et al., "Conditional Variational Autoencoder
    with Adversarial Learning for End-to-End Text-to-Speech," in ICML, 2021.

    Args:
        value (np.ndarray): Log-likelihood matrix of shape (text_length, latent_variable_length).
            The (i, j)-th entry contains the log-likelihood of the j-th latent variable
            for the given i-th prior mean and variance:
            ``value[i, j] = log N(f(z)_j; mu_i, sigma_i)``.

    Returns:
        np.ndarray: Most likely alignment of shape (text_length, latent_variable_length).
    """
    t_x, t_y = value.shape  # [text_length, letent_variable_length]
    path = np.zeros((t_x, t_y))

    # A cache to store the log-likelihood for the most likely alignment so far.
    Q = np.full((t_x, t_y), -np.inf)

    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if y == 0:  # Base case. If y is 0, the possible x value is only 0.
                Q[x, 0] = value[x, 0]
            else:
                if x == 0:
                    v_prev = -np.inf
                else:
                    v_prev = Q[x - 1, y - 1]
                v_cur = Q[x, y - 1]
                Q[x, y] = value[x, y] + max(v_prev, v_cur)

    # Backtrack from last observation.
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or Q[index, y - 1] < Q[index - 1, y - 1]):
            index = index - 1

    return path
