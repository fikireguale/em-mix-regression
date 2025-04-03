'''
# Python 3.11.9 64-bit version in usr/local/bin/python3
nj: number of measurements for person j: fixed for now
M = number of individuals (trajectories)
K = max number of clusters = number of polynomials: fixed for now?
'''
import numpy as np
import copy
import matplotlib.pyplot as plt
import statsmodels.api as sm # to test mStepLibrary
import scipy.stats as st
from copy import deepcopy

def generateH(resp, M, K, nj):
    # Generate the H matrix (N x N) for each cluster
    Hk_matrices = []
    for k in range(K):
        weights = []
        for j in range(M):
            temp = [resp[j][k]] * nj
            weights += temp
        Hk_matrices.append(np.diag(weights))
    return Hk_matrices

def generateData(M, K, nj, l, noise_mean, noise_std, seed=42):
    """
    """
    # Generate X matrix
    ones = np.array([1]*nj)
    firstOrder = np.array([1, 6, 9, 12, 21, 26, 36, 39, 43, 44])
    secondOrder = np.square(firstOrder)
    X = np.column_stack((ones, firstOrder, secondOrder))
    XCopy = copy.deepcopy(X)
    for i in range(1, M):
        X = np.vstack((X, XCopy))

    # Generate Y matrix
    poly1 = lambda x: 120 + 4*x 
    poly2 = lambda x: 10 + 2*x + 0.1*(x**2)
    poly3 = lambda x: 250 - 0.75*x
    polyList = [poly1, poly2, poly3]

    Y = np.array([])
    i = 0
    for poly in polyList:
        for _ in range(l): # For "l" number of people/trajectories
            Yj = np.array([]) # Each person/trajectory is a Yj vector
            for _ in range(nj): # Sample nj points
                Yj = np.append(Yj, poly(X[:,1][i])) # array[:, 0] --> get 2nd column
                i += 1
            Y = np.append(Y, Yj)
    #np.random.seed(seed)
    #noise = np.random.normal(noise_mean, noise_std, Y.shape)
    #Y += noise

    # Generate responsibilities == membership_probs (M x K), Note: a compact Hk matrix
    resp = np.random.rand(M, K)
    resp /= resp.sum(axis=1, keepdims=True)  # Normalize

    # Generate the H matrix (N x N) for each cluster
    Hk_matrices = generateH(resp, M, K, nj)
    '''
    Hk_matrices = []
    for k in range(K):
        weights = []
        for j in range(M):
            temp = [resp[j][k]] * nj
            weights += temp
        Hk = np.diag(weights)
        Hk_matrices.append(Hk) # Hk_matrices = [(NxN matrix1), ..., (NxN matrixK)]
    '''
    return X, Y, Hk_matrices, resp

def sumHjk(Hk, nj):
    # Hk must be (N x N)
    diagonal_sum = 0
    for i in range(0, Hk.shape[0], nj):
        diagonal_sum += Hk[i, i]
    return diagonal_sum

def mStep(X, Y, H, M, K, nj):
    ''' Estimate regression parameters for each cluster'''
    params = []
    for k in range(K):
        Hk = H[k]

        # Compute beta_hat
        Xt_H_X = X.T @ Hk @ X
        Xt_H_Y = (X.T @ Hk @ Y).reshape(-1,1)
        beta_hat = np.linalg.inv(Xt_H_X) @ Xt_H_Y

        # Compute variance
        sum_hjk = sumHjk(Hk, nj)
        residuals = Y.reshape(-1,1) - (X @ beta_hat)
        sigma2_hat = residuals.T @ Hk @ residuals
        sigma2_hat = sigma2_hat / sum_hjk

        # Compute mixing weights
        w_hat = (1/M) * sum_hjk

        # Cleanup
        beta_hat = beta_hat.T[0]
        sigma2_hat = sigma2_hat[0][0]
        params.append([beta_hat, sigma2_hat, w_hat])
        '''
        # Display results
        print("No Library")
        print("For k =", k)
        print("β̂_k       =", beta_hat)
        print("σ̂²_k      =", sigma2_hat)
        print("ŵ_k       =", w_hat)
        '''
    return params

def plot(x, y):
    plt.scatter(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Function value')
    plt.title('')
    plt.legend()
    plt.show()
    return

def mStepLibrary(X, Y, H, M, K, nj):
    params = []
    for k in range(K):
        Hk = H[k]

        # Weighted Least Squares
        model = sm.WLS(Y, X, weights=np.diag(Hk))
        results = model.fit()

        # --- Extract estimates ---
        beta_hat = results.params                             
        sigma2_hat = results.scale
        # Note: Statsmodel WLS returns the unbiased estimate for variance, not MLE. 
        
        w_hat = (1/M) * sumHjk(Hk, nj)                       
        params.append([beta_hat, sigma2_hat, w_hat])
        '''
        # Display results
        print()
        print("Library")
        print("For k =", k)
        print("β̂_k       =", beta_hat)
        print("σ̂²_k      =", sigma2_hat)
        print("ŵ_k       =", w_hat)
        '''
    return params

def test_WLS():
    # Global test variables
    RandomSeed = 42
    np.random.seed(RandomSeed)
    M = 2
    nj = 2
    K = 1

    X = np.array([
    [1, 1],
    [1, 2],
    [1, 1],
    [1, 2]
    ])

    Y = np.array([
        2.0,
        4.1,
        1.2,
        2.1
    ])

    # Two trajectories: [2.0, 4.1] and [1.2, 2.1]
    # Soft membership of 0.8 for traj 1 and 0.2 for traj 2
    Hk = np.diag([
        0.8, 0.8,  # for first trajectory
        0.2, 0.2   # for second trajectory
    ])
    H = [Hk]

    [[beta, sigma, w_hat]] = mStep(X, Y, H, M, K, nj)

    passed = (
        np.allclose(beta, [-0.02, 1.86]) and
        np.isclose(sigma, 0.7424) and
        np.isclose(w_hat, 0.5)
    )
    print("test_WLS() passed:", passed)
    return passed

def gaussian_likelihood(y, x, beta_k, sigma2_k):
    """
    Computes the Gaussian likelihood for a single trajectory's observations 
    under component k, given regression parameters.

    Args:
        y (np.ndarray): Observed outputs, shape (n_j,)
        x (np.ndarray): Design matrix X_j, shape (n_j, p+1)
        beta_k (np.ndarray): Regression coefficients, shape (p+1,)
        sigma2_k (float): Variance

    Returns:
        float: Likelihood of y under N(x^T beta_k, sigma2_k)
    """
    mu = x.reshape(-1,1) @ beta_k                     # shape (n_j,)
    #sigma = np.sqrt(sigma2_k)
    pdf_vals = norm.pdf(y, loc=mu, scale=sigma2_k)
    return np.prod(pdf_vals)           # product over i

def eStep(X, Y, params, M, K, nj):
    '''Update membership probabilities:
        hjk = Calculate likelihood of all trajectory j's points under cluster k
    '''
    resp = np.zeros((M, K)) # responsabilities
    F = np.zeros((K, M, nj))
    for k in range(K):
        [beta, sigma, w_hat] = params[k]
        for j in range(M): # For every person in cluster k
            start = j*nj
            end = start + nj
            yj = Y[start:end]
            xj = X[start:end]
            for i in range(nj):
                mu = xj[i] @ beta
                fk = st.norm.pdf(yj[i], loc=mu, scale=sigma)
                F[k][j][i] = fk
            resp[j][k] = w_hat * np.prod(F[k][j])
    total = 0
    for k in range(K):
        w = params[k][2]
        total += w * np.sum(F[k])

    if total > 0:
        resp /= total
    else:
        print("Dividing by <= 0")

    test_resp = deepcopy(resp)
    resp /= resp.sum(axis=1, keepdims=True)  # Normalize

    # Generate H matrix from membership probs (resp)
    H = generateH(resp, M, K, nj)
    return H, test_resp

def test_eStep_resp():
    # Global test variables
    RandomSeed = 42
    np.random.seed(RandomSeed)
    M = 2
    nj = 2
    K = 1

    X = np.array([
    [1, 1],
    [1, 2],
    [1, 1],
    [1, 2]
    ])

    Y = np.array([
        2.0,
        4.1,
        1.2,
        2.1
    ])

    # Two trajectories: [2.0, 4.1] and [1.2, 2.1]
    # Soft membership of 0.8 for traj 1 and 0.2 for traj 2
    Hk = np.diag([
        0.8, 0.8,  # for first trajectory
        0.2, 0.2   # for second trajectory
    ])
    H = [Hk]

    params = mStep(X, Y, H, M, K, nj)

    '''
    params = 
        beta: [-0.02, 1.86]
        sigma: 0.7424
        w_hat: 0.5
    '''
    H, test_resp = eStep(X, Y, params, M, K, nj)

    passed = (
        np.allclose(test_resp, np.array([[0.17268546], [0.01381672]]))
    )
    print("test_eStep_resp() passed:", passed)
    return passed

def test_eStep_poly_match():
    import numpy as np
    from scipy.stats import norm

    np.random.seed(42)
    M = 2
    nj = 2
    K = 2

    # Design matrix for two people, each with x = [1, 2]
    # Columns: [1, x, x^2]
    X = np.array([
        [1, 1, 1],
        [1, 2, 4],
        [1, 1, 1],
        [1, 2, 4]
    ])

    # Response Y:
    # Person 1 (first two values) follows y = 2x → [2, 4]
    # Person 2 (last two values) follows y = 1 + x^2 → [2, 5]
    Y = np.array([
        2.0, 4.0,   # person 1
        2.0, 5.0    # person 2
    ])

    # True generating functions:
    # Cluster 0: y = 2x → beta = [0, 2, 0]
    # Cluster 1: y = 1 + x^2 → beta = [1, 0, 1]
    params = [
        [np.array([0.0, 2.0, 0.0]), 0.01, 0.5],  # Very small variance → confident
        [np.array([1.0, 0.0, 1.0]), 0.01, 0.5]
    ]

    # Call E-step
    H, _ = eStep(X, Y, params, M, K, nj)

    # Should show: person 1 assigned to cluster 0, person 2 to cluster 1
    passed = np.all(H == np.diag([1,1,0,0,0,0,1,1])) # expected
    print("test_eStep_poly_match passed:", passed)
    return passed

#def initialize(X, Y, M, K, nj, l, noise_mean, noise_std, seed=42):

def fit2(X, Y, M, K, nj, l, noise_mean, noise_std, seed=42, max_iter=10, tol=0.001):
    a = initialize(X, Y, M, K, nj, l, noise_mean, noise_std, seed=42)
    for i in range(max_iter):
        old_means = np.copy(self.means)
        responsibilities = self._e_step(X)
        self._m_step(X, responsibilities)

        if np.sum((self.means - old_means)**2) < tol:
            break
    return

def fit(X, Y, H, M, K, nj):
    for i in range(max_iter):
        params = mStep(X, Y, H, M, K, nj)
        H, _ = eStep(X, Y, params, M, K, nj)

        if np.sum((self.means - old_means)**2) < tol:
            break
    return

def main():
    #plot(X[:,1], Y)
    #test_WLS()
    #test_eStep_resp()
    #test_eStep_poly_match()
    
    M, K, nj, l, noise_mean, noise_std, max_iter = 12, 3, 10, 4, 0, 10, 5
    X, Y, H, resp = generateData(M, K, nj, l, noise_mean, noise_std)
    
    # Instead, of mstep estep use fit()
    params = mStep(X, Y, H, M, K, nj) # params1, ..., paramsK
    H, _ = eStep(X, Y, params, M, K, nj)
    
    #fit(X, Y, H, M, K, nj, max_iter)

    

    

    return
main()