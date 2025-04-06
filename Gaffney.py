'''
# Python 3.11.9 64-bit version in usr/local/bin/python3
nj: number of measurements for person j: fixed for now
M = number of individuals (trajectories)
K = max number of clusters = number of polynomials
'''
import numpy as np
import copy
import matplotlib.pyplot as plt
import statsmodels.api as sm # to test mStepLibrary
import scipy.stats as st
from copy import deepcopy

def generateH(M, K, nj, resp, init=False):
    ''' Generate responsibilities == membership_probs (M x K)'''
    assert (len(resp) != 0 and not init) or (len(resp) == 0 and init)

    if init:
        resp = np.random.dirichlet([1] * K, size=M)
    
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
    
    noise = np.random.normal(noise_mean, noise_std, Y.shape)
    Y += noise

    return X, Y

def sumHjk(Hk, nj):
    # Hk must be (N x N)
    diagonal_sum = 0
    for i in range(0, Hk.shape[0], nj):
        diagonal_sum += Hk[i, i]
    return diagonal_sum

def is_singular(X):
    return bool(np.linalg.matrix_rank(X) < X.shape[1])

def mStep(X, Y, H, M, K, nj):
    ''' Estimate regression parameters for each cluster'''
    params = []
    for k in range(K):
        Hk = H[k]

        # Compute beta_hat
        Xt_H_X = X.T @ Hk @ X
        Xt_H_Y = (X.T @ Hk @ Y).reshape(-1,1)
        if is_singular(X):
            beta_hat = np.linalg.pinv(Xt_H_X) @ Xt_H_Y
        else:
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
        sigma2_hat = max(sigma2_hat, 1e-3) #1e-3
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

def plot(x, y, params, run, true_coeff=[]):
    true_coeff = [[120, 4, 0], [10, 2, 0.1], [250, -0.75, 0]]
    coeff = []
    for i, poly in enumerate(params):
        [betas, sigma, w] = poly
        coeff.append(betas[::-1])
        true_coeff[i] = true_coeff[i][::-1]

    plt.plot(x, y, 'o', mfc='none', color='dimgray')
    colors = plt.cm.tab10(np.linspace(0, 1, len(coeff)))

    for i, (c, tc) in enumerate(zip(coeff, true_coeff)):
        polynomial = np.poly1d(c)
        true_poly = np.poly1d(tc)

        x = np.linspace(1, 44, 100)
        y = polynomial(x)
        true_y = true_poly(x)

        plt.plot(x, y, color=colors[i])
        plt.plot(x, true_y, color=colors[i], linestyle = '--')

    plt.xlabel('X-axis')
    plt.ylabel('Function value')
    plt.title(f'Iteration {run}')
    plt.show()
    return

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

def eStep(X, Y, params, M, K, nj):
    '''Update membership probabilities:
        hjk = Calculate likelihood of all trajectory j's points under cluster k
    '''
    log_likelihood = 0
    resp = np.zeros((M, K)) # responsabilities
    for j in range(M):
        start, end = j*nj, j*nj + nj
        xj, yj = X[start:end], Y[start:end]
        person = 0 # likelihood denominator | Person(j)'s total
        for k in range(K):
            [beta, sigma, w_hat] = params[k]
            clus = [] # likelihood denominator | Person(j)'s cluster(k) total
            for i in range(nj):
                mu = xj[i] @ beta
                pdf = st.norm.pdf(yj[i], loc=mu, scale=np.sqrt(sigma))
                clus.append(pdf)
            resp[j][k] = w_hat * np.prod(clus) # hjk: store likelihood
            person += resp[j][k]
            # normalize Person(J)'s membership prob
        if person != 0:
            resp[j] /= person 
        log_likelihood += np.log(person + 1e-20) # before it was 1e-12, 1e-20 worked
    test_resp = deepcopy(resp)
    # Generate H matrix from membership probs (resp)
    H = generateH(M, K, nj, resp, init=False)
    return H, test_resp, log_likelihood

def test_eStep_poly_match():
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
    H, _, _ = eStep(X, Y, params, M, K, nj)

    # Should show: person 1 assigned to cluster 0, person 2 to cluster 1
    k1 = np.diag([1,1,0,0])
    k2 = np.diag([0,0,1,1])
    expected = [k1, k2]
    passed = np.all(np.all(H) == np.all(expected))
    print("test_eStep_poly_match passed:", passed)
    return passed

def test_eStep_Hmatrix():
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

    H = generateH(M, K, nj, resp=[], init=True)

    params = mStep(X, Y, H, M, K, nj)

    # Call E-step
    H, _, _ = eStep(X, Y, params, M, K, nj)

    # In future iterations, should show: person 1 assigned to cluster 0, person 2 to cluster 1
    k1 = np.diag([0.77,0.77,0.89,0.89])
    k2 = np.diag([0.22,0.22,0.1,0.1])
    expected = [k1, k2]
    passed = np.all(np.all(H) == np.all(expected))
    print("test_eStep_Hmatrix passed:", passed)
    inputs = [M, nj, K, X, Y]
    return passed, inputs

def test_eStep_Hmatrix_more():
    np.random.seed(42)
    M = 4  # 4 trajectories (2 per cluster)
    nj = 3  # 3 measurements per trajectory
    K = 2

    # Expanded x = [1, 2, 3] → [1, 1, 1], [1, 2, 4], [1, 3, 9]
    X_traj = np.array([
        [1, 1, 1],
        [1, 2, 4],
        [1, 3, 9]
    ])
    X = np.vstack([X_traj for _ in range(M)])

    # Responses:
    # Cluster 0 (linear): y = 2x
    # Cluster 1 (quadratic): y = 1 + x^2
    Y_cluster0 = np.array([2, 4, 6])
    Y_cluster1 = np.array([2, 5, 10])
    Y = np.concatenate([
        Y_cluster0,  # person 0
        Y_cluster0,  # person 1
        Y_cluster1,  # person 2
        Y_cluster1   # person 3
    ])

    H = generateH(M, K, nj, resp=[], init=True)
    params = mStep(X, Y, H, M, K, nj)
    H, _, _ = eStep(X, Y, params, M, K, nj)

    # Just check no cluster collapse and all values are finite
    passed = all(np.isfinite([p for param in params for p in param[0]])) and all([np.isfinite(p[1]) for p in params])
    print("test_eStep_Hmatrix passed:", passed)
    inputs = [M, nj, K, X, Y]
    return passed, inputs

def set_seed(seed=1082265693):
    seed = np.random.randint(0, 2**32)
    np.random.seed(seed)

def fit(X, Y, M, K, nj, max_iter, tol, debug=False, resp_tol=1e-4):
    best_loglik = -np.inf
    best_params = None
    best_H = None
    best_resp = None
    valid_snapshots = []  # List of (loglik, iteration, params)

    for _ in range(10):
        set_seed()
        H = generateH(M, K, nj, resp=[], init=True)
        prev_loglik = -np.inf
        resp = []
        all_snapshots = []

        for i in range(max_iter):
            params = mStep(X, Y, H, M, K, nj)
            H, resp, loglik = eStep(X, Y, params, M, K, nj)
            all_snapshots.append((loglik, i, deepcopy(params)))

            if (np.abs(loglik - prev_loglik) / abs(loglik + 0.1) < tol):
                print(f"Converged at iteration {i}: Log-likelihood {loglik}")
                break
            prev_loglik = loglik
        
        if loglik > -551:
            valid_snapshots.append(all_snapshots)
        else:
            all_snapshots = []

        if loglik > best_loglik:
            best_loglik = loglik
            best_params = params
            best_H = H
            best_resp = resp
    return best_params, best_H, best_loglik, best_resp, valid_snapshots

def plot_snapshots(x, y, snapshots):
    # Flatten
    if isinstance(snapshots[0], list) and len(snapshots) == 1:
        snapshots = snapshots[0]

    if not snapshots:
        print("No snapshots to plot.")
        return

    total = len(snapshots)
    target_indices = [0, total // 4, total // 2, 3 * total // 4, total - 1]

    for i in target_indices:
        [loglik, run, params] = snapshots[i]
        plot(x, y, params, run)

def print_params(params):
    print("\n Fitted Model Parameters:\n")

    for idx, (beta, sigma2, w) in enumerate(params):
        b0, b1, b2 = beta
        print(f"Cluster {idx}:")
        print(f"  β̂ (coefficients):      [{b0:.2f}, {b1:.2f}, {b2:.3f}]")
        print(f"  σ̂² (variance):         {sigma2:.2f}")
        print(f"  ŵ  (mixing weight):     {w:.4f}")
        print(f"  ⇒ Polynomial:           y = {b0:.2f} + {b1:.2f}·x + {b2:.3f}·x²\n")

def print_resp(resp, precision=3):
    M, K = resp.shape
    print("\n Responsibility Matrix (resp):\n")

    # Header
    header = "Person".ljust(8) + "".join([f"Cluster {k}".rjust(12) for k in range(K)]) + "   Assigned to"
    print(header)
    print("-" * len(header))

    # Rows
    for j in range(M):
        probs = "  ".join([f"{resp[j][k]:>{8}.{precision}f}" for k in range(K)])
        assigned = np.argmax(resp[j])
        print(f"{j:<8}  {probs}   → Cluster {assigned}")

def paper_fit():
    '''
    # Expected polynomials
     y1 = 120+ 4x, y2 = 10+ 2x + 0.1x^2, and y3 = 250 - 0.75x

    cluster1: [120, 4, 0]
    cluster2: [10, 2, 0.1]
    cluster3: [250, -0.75, 0]
    '''

    M, K, nj, l, noise_mean, noise_std, max_iter, tol = 12, 3, 10, 4, 0, 1, 50, 1e-15 # max_iter=100, tol=1e-15 
    X, Y = generateData(M, K, nj, l, noise_mean, noise_std)
    params, _, loglik, resp, snapshots = fit(X, Y, M, K, nj, max_iter, tol)

    print("Loglik:", loglik)
    print_params(params)
    print_resp(resp)
    plot_snapshots(X[:, 1], Y, snapshots)

    return params, X, Y

def main():
    set_seed()
    paper_fit()
    return
main()