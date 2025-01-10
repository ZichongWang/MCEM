import numpy as np
from scipy.special import expit #logistic sigmoid function
import time
from multiprocessing import Pool
from joblib import Parallel, delayed
from numba import njit, prange
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_fixed_X(n, T):
    X = np.random.normal(loc=0, scale=1, size=(n, T))
    X = X.clip(min=-3, max=3) # restrict $X$ to $[-3, 3]$
    return X

def generate_data(X, beta1, beta2, sigma1, sigma2, pi1):
    start_time = time.time()
    n = X.shape[0]
    T = X.shape[1]

    U = 2 - np.random.binomial(1, pi1, size=n)# sample from bernoulli distribution to decide cluster membership
    print(f"U.mean(): {U.mean()}")

    Z1 = np.random.normal(loc=0, scale=sigma1, size=n)
    Z2 = np.random.normal(loc=0, scale=sigma2, size=n)

    # Initialize Y
    Y = np.zeros((n, T))

    # Generate Y

    P = np.where(U[:, None] == 1, 
             expit(beta1 * X + Z1[:, None]), 
             expit(beta2 * X + Z2[:, None]))
    Y = np.random.binomial(1, P)
    # for i in range(n):
    #     for j in range(T):
    #         if U[i] == 1:
    #             # Conditional expectation of group 1
    #             P_ij = np.exp(beta1 * X[i, j] + Z1[i]) / (1 + np.exp(beta1 * X[i, j] + Z1[i]))
    #         else:
    #             # Conditional expectation of group 2
    #             P_ij = np.exp(beta2 * X[i, j] + Z2[i]) / (1 + np.exp(beta2 * X[i, j] + Z2[i]))
            
    #         # Generate Y_ij from Bernoulli distribution
    #         Y[i, j] = np.random.binomial(1, P_ij)
    print(f"Time of data generation: {time.time() - start_time}")
    return Y


def inv_logit(x):
    # sigmoid function, returns $1 / (1 + \exp(-x))$
    return expit(x)

def f_c_Z(Z, sigma):
    '''
    Given the cluster assignment $c$, compute the density of $Z$.
    params:
    Z: np.array, $Z \in \mathbb{R}$
    sigma: float, standard deviation of the normal distribution, $\sigma \in \mathbb{R}$

    return:
    np.array, density of $Z$
    '''
    term1 = 1 / (sigma * np.sqrt(2 * np.pi))
    term2 = np.exp(-Z**2 / (2 * sigma**2))
    return term1 * term2

def f_c_Y(Y, X, Z, beta, is_log=False):
    '''
    Compute $\prod_{j=1}^T f_c(Y_{ij} \mid Z_i^{(k)}, \Omega^{(m)})$ of a sample.
    Log-Exp is used to prevent numerical underflow.
    param:
    Y: np.array, $Y \in \mathbb{R}^{1 \times T}$
    X: np.array, $X \in \mathbb{R}^{1 \times T}$
    Z: np.array, $Z \in \mathbb{R}$
    beta: float, $\beta \in \mathbb{R}$
    '''
    linear_pred = beta * X + Z
    P_ij = inv_logit(linear_pred)
    P_ij = np.clip(P_ij, 1e-10, 1 - 1e-10) # numerical stability
    log_likelihood = np.sum(Y * np.log(P_ij) + (1 - Y) * np.log(1 - P_ij))
    if is_log:
        return log_likelihood
    return np.exp(log_likelihood)


def compute_A_c(Y, X, Z, beta, sigma):
    # Compute $A_c(Z_i^{(k)}, Y_{ij}, \Omega^{(m)})= f_c(Z_i^{(k)} \mid \sigma_c^{(m)}) \left[\prod_{j=1}^T f_c(Y_{ij} \mid Z_i^{(k)}, \Omega^{(m)}) \right]$
    Z_likelihood = f_c_Z(Z, sigma)
    Y_likelihood = f_c_Y(Y, X, Z, beta)
    return Z_likelihood * Y_likelihood

def compute_posterior_pi1(Y, X, Z, beta1, beta2, sigma1, sigma2, pi1):
    # Compute the posterior probability of $\widetilde{\pi}_{i1} = \frac{\pi_1^{(m)} \cdot A_1(Z_i^{(k)}, Y_{ij},\Omega^{(m)})}{\pi_1^{(m)} \cdot A_1(Z_i^{(k)}, Y_{ij},\Omega^{(m)}) + \pi_2^{(m)} \cdot A_2(Z_i^{(k)}, Y_{ij},\Omega^{(m)})}$
    A_1 = compute_A_c(Y, X, Z, beta1, sigma1)
    A_2 = compute_A_c(Y, X, Z, beta2, sigma2)
    if np.isnan(A_1) or np.isnan(A_2):
        print('A_1 or A_2 is NaN')
        return 0.5
    pi_tilde_i1 = pi1 * A_1 / (pi1 * A_1 + (1 - pi1) * A_2)
    if np.isnan(pi_tilde_i1):
        print('pi is NaN')
        return 0.5
    return pi_tilde_i1

def generate_w(pi_tilde_i1):
    # Generate $w_i^{(k+1)} \sim \text{Bernoulli}(\widetilde{\pi}_{i1})$
    if np.isnan(pi_tilde_i1):
        raise ValueError('pi is NaN')
    if pi_tilde_i1 > 1 or pi_tilde_i1 < 0:
        raise ValueError(f'Invalid value of pi: {pi_tilde_i1}, out of range')
    pi_tilde_i1 = np.clip(pi_tilde_i1, 1e-2, 1-1e-2) # numerical stability
    w_i = 1 - np.random.binomial(1, pi_tilde_i1) + 1 # Opposite of the Bernoulli distribution
    return w_i

def sampling_w(X, Y, Z, beta1, beta2, sigma1, sigma2, pi1):
    '''
    Given the parameters of last iteration (pi, Z, beta, sigma, w), sample w.
    '''
    pi_tilde_i1 = compute_posterior_pi1(Y, X, Z, beta1, beta2, sigma1, sigma2, pi1)
    w_i = generate_w(pi_tilde_i1)
    return w_i



def generate_Z_star(sigma1, sigma2, w_i):
    # Generate $Z_i^{(k+1)} \sim \mathcal{N}(\mu_{w_i}, \sigma_{w_i})$
    if w_i == 1:
        Z_star = np.random.normal(0, sigma1)
    else:
        Z_star = np.random.normal(0, sigma2)
    return Z_star

def calculate_acceptance_rate(Y, X, Z, Z_star, w_i, beta1, beta2, sigma1, sigma2):
    # Calculate $\alpha^*=\min \left\{ 1,\quad \frac{\prod\limits_{j=1}^{T}f_c(Y_{ij}\mid Z_i^*, w_{ic}^{(k+1)}, \Omega^{(m)})}{\prod\limits_{j=1}^{T}f_c(Y_{ij}\mid Z_i^{(k)}, w_{ic}^{(k+1)}, \Omega^{(m)})} \right\}$
    if w_i == 1:
        beta = beta1
        sigma = sigma1
    elif w_i == 2:
        beta = beta2
        sigma = sigma2
    else:
        raise ValueError(f"Invalid cluster assignment w_i: {w_i}. Expected 1 or 2.")
    # Z_star = Z + np.random.normal(0, 0.5)
    likelihood_proposed = f_c_Y(Y, X, Z_star, beta)
    likelihood_proposed = max(likelihood_proposed, 1e-10)
    likelihood_current = f_c_Y(Y, X, Z, beta)
    likelihood_current = max(likelihood_current, 1e-10)
    
    # prior_ratio = f_c_Z(Z_star, sigma) / f_c_Z(Z, sigma)

    acceptance_rate = likelihood_proposed / likelihood_current
    return min(1, acceptance_rate)

def generate_next_z(Z_star, Z, acceptance_rate):
    # Generate $Z_i^{(k+1)}$ by Metropolis-Hastings algorithm
    u = np.random.uniform(0, 1)
    if u < acceptance_rate:
        Z = Z_star
    return Z

def sampling_Z(Y, X, Z, w_i, beta1, beta2, sigma1, sigma2):
    '''
    Given the parameters of last iteration (Z, beta, sigma) and w from this iter, sample Z.
    '''
    Z_star = generate_Z_star(sigma1, sigma2, w_i)
    acceptance_rate = calculate_acceptance_rate(Y, X, Z, Z_star, w_i, beta1, beta2, sigma1, sigma2)
    Z = generate_next_z(Z_star, Z, acceptance_rate)
    return Z

def sample_chain(i, Y_i, X_i, Z_i, beta1, beta2, sigma1, sigma2, pi1, K=500, k=100):
    '''
    Sample the chain for a single sample.
    '''
    w = 0
    Z = Z_i
    sampled_w = []
    sampled_Z = []
    
    for iteration in range(K):
        w = sampling_w(X_i, Y_i, Z, beta1, beta2, sigma1, sigma2, pi1)
        Z = sampling_Z(Y_i, X_i, Z, w, beta1, beta2, sigma1, sigma2)
        
        if iteration >= k:
            sampled_w.append(w)
            sampled_Z.append(Z)
    return (i, sampled_w, sampled_Z)

def E_step(Y, X, Z, beta1, beta2, sigma1, sigma2, pi1, K=500, k=100, n_jobs=-1):
    """
    Perform the E-step of the MCEM algorithm, updating w and Z for all samples.
    
    Args:
        Y: Observed responses (n x T), binary values {0, 1}
        X: Fixed effects (n x T)
        Z: Current values of Z (n x 1)
        beta1, beta2: Regression coefficients for two clusters
        sigma1, sigma2: Standard deviations for two clusters
        pi1: Prior probability for cluster 1
        K: Number of iterations for sampling w and Z
        k: Number of burn-in iterations

    Returns:
        w: Updated cluster assignments (n x 1)
        Z: Updated random effects (n x 1)
    """
    n = Y.shape[0]  # Number of samples
    results = Parallel(n_jobs=n_jobs)(
        delayed(sample_chain)(
            i,
            Y[i],
            X[i],
            Z[i],
            beta1,
            beta2,
            sigma1,
            sigma2,
            pi1,
            K,
            k
        ) for i in range(n)
    )

    all_w = np.zeros((K - k, n), dtype=int)
    all_Z = np.zeros((K - k, n)) 
    for result in results:
            i, sampled_w, sampled_Z = result
            all_w[:, i] = sampled_w  # 将采样结果放在第 i 列
            all_Z[:, i] = sampled_Z
        
    return all_w, all_Z


def updating_pai_c(all_w):
    '''
    Calculate $\pi_c = \frac{\sum_{i=1}^n \sum_{k=1}^K \omega_{ic}^{(k)}}{n \cdot K}, \quad c = 1, 2$
    '''
    K, n = all_w.shape
    pi1 = np.mean(all_w == 1)
    pi2 = np.mean(all_w == 2)
    return pi1, pi2



def updating_beta_c(X, Y, all_w, all_Z, lr=0.01, max_iter=100, tol=1e-4,update=True):
    """
    Perform SGD optimization for two groups (beta1 and beta2) with Monte Carlo samples.
    
    Parameters:
        X (ndarray): Design matrix, shape (n, T).
        Y (ndarray): Response matrix, shape (n, T).
        all_w (ndarray): Indicator function from E-step, shape (K, n), values are 1 or 2.
        all_Z (ndarray): Random effects from E-step, shape (K, n).
        beta1_init (float): Initial beta for group 1.
        beta2_init (float): Initial beta for group 2.
        lr (float): Learning rate for SGD.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
    
    Returns:
        beta1 (float): Final beta for group 1.
        beta2 (float): Final beta for group 2.
    """
    if not update:
        return 1, 1.5
    K, n = all_w.shape  # K: number of Monte Carlo samples, n: number of data samples
    T = X.shape[1]  # Number of observations per sample
    
    # Initialize beta values
    beta1 = np.random.normal(1, 1)  # Initial beta for group 1
    beta2 = np.random.normal(1, 1)  # Initial beta for group 2

    X_expanded = X[np.newaxis, :, :]  # Shape: (1, n, T)
    Y_expanded = Y[np.newaxis, :, :]  # Shape: (1, n, T)
    
    for iter_num in range(max_iter):
        # Compute mask for group 1 and group 2
        mask1 = (all_w == 1)  # Shape: (K, n)
        mask2 = (all_w == 2)  # Shape: (K, n)
        
        # Expand masks and Z for broadcasting
        # mask1 and mask2: (K, n) -> (K, n, 1)
        mask1_expanded = mask1[:, :, np.newaxis]  # Shape: (K, n, 1)
        mask2_expanded = mask2[:, :, np.newaxis]  # Shape: (K, n, 1)
        
        # Z: (K, n) -> (K, n, 1)
        Z_expanded = all_Z[:, :, np.newaxis]      # Shape: (K, n, 1)

        # Compute linear predictors
        # For group 1: eta1 = beta1 * X + Z
        eta1 = beta1 * X_expanded + Z_expanded   # Shape: (K, n, T)
        # For group 2: eta2 = beta2 * X + Z
        eta2 = beta2 * X_expanded + Z_expanded   # Shape: (K, n, T)
        
        # Compute logistic probabilities
        P1 = 1 / (1 + np.exp(-eta1))             # Shape: (K, n, T)
        P2 = 1 / (1 + np.exp(-eta2))             # Shape: (K, n, T)

        # Compute gradients
        # grad1 = sum over K, n, T: mask1 * (Y - P1) * X
        # grad2 = sum over K, n, T: mask2 * (Y - P2) * X
        # Utilize element-wise operations and sum over K and n
        error1 = (Y_expanded - P1) * X_expanded   # Shape: (K, n, T)
        error2 = (Y_expanded - P2) * X_expanded   # Shape: (K, n, T)
        
        grad1 = np.sum(mask1_expanded * error1) / K  # Scalar
        grad2 = np.sum(mask2_expanded * error2) / K  # Scalar
            
        # Update beta values
        beta1_new = beta1 + lr * grad1
        beta2_new = beta2 + lr * grad2
        # Check convergence
        if max(abs(beta1_new - beta1), abs(beta2_new - beta2)) < tol:
            beta1, beta2 = beta1_new, beta2_new
            print(f"Converged at iteration {iter_num + 1}")
            break
        
        # Update beta values for next iteration
        beta1, beta2 = beta1_new, beta2_new
    
    return beta1, beta2


@njit(parallel=True)
def compute_gradients_numba(beta1, beta2, X, Y, all_w, all_Z, K, n, T):
    grad1 = 0.0
    grad2 = 0.0
    for k in prange(K):
        for i in range(n):
            for t in range(T):
                eta = 0.0
                P = 0.0
                if all_w[k, i] == 1:
                    eta = beta1 * X[i, t] + all_Z[k, i]
                    P = 1.0 / (1.0 + np.exp(-eta))
                    grad1 += (Y[i, t] - P) * X[i, t]
                elif all_w[k, i] == 2:
                    eta = beta2 * X[i, t] + all_Z[k, i]
                    P = 1.0 / (1.0 + np.exp(-eta))
                    grad2 += (Y[i, t] - P) * X[i, t]
    grad1 /= K
    grad2 /= K
    return grad1, grad2

def updating_beta_c_numba(X, Y, all_w, all_Z, lr=0.01, max_iter=100, tol=1e-4, update=True):
    """
    Perform optimization for two groups (beta1 and beta2) with Monte Carlo samples using Numba for acceleration.
    
    Parameters:
        X (ndarray): Design matrix, shape (n, T).
        Y (ndarray): Response matrix, shape (n, T).
        all_w (ndarray): Indicator function from E-step, shape (K, n), values are 1 or 2.
        all_Z (ndarray): Random effects from E-step, shape (K, n).
        lr (float): Learning rate for SGD.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
        update (bool): Whether to perform update. If False, return default values.
    
    Returns:
        beta1 (float): Final beta for group 1.
        beta2 (float): Final beta for group 2.
    """
    if not update:
        return 1.0, 1.5
    
    K, n = all_w.shape
    T = X.shape[1]
    
    # Initialize beta values
    beta1 = 0.0
    beta2 = 0.0
    
    for iter_num in range(max_iter):
        # Compute gradients using Numba-accelerated function
        grad1, grad2 = compute_gradients_numba(beta1, beta2, X, Y, all_w, all_Z, K, n, T)
        
        # Update beta values
        beta1_new = beta1 + lr * grad1
        beta2_new = beta2 + lr * grad2
        
        if iter_num == max_iter//2:
            lr /= 10
        # Check convergence
        if max(abs(beta1_new - beta1), abs(beta2_new - beta2)) < tol:
            beta1, beta2 = beta1_new, beta2_new
            print(f"Numba: Converged at iteration {iter_num + 1}")
            break
        
        # Update beta values for next iteration
        beta1, beta2 = beta1_new, beta2_new
    
    return beta1, beta2

def compute_sigma(all_w, all_Z):
    # Mask for cluster 1
    cluster_1_mask = (all_w == 1)
    sigma_1_numerator = np.sum(cluster_1_mask * all_Z**2)  # Sum of Z^2 for cluster 1
    sigma_1_denominator = np.sum(cluster_1_mask)          # Count of samples in cluster 1
    sigma_1_squared = sigma_1_numerator / sigma_1_denominator if sigma_1_denominator > 0 else 0
    sigma_1 = np.sqrt(sigma_1_squared)

    # Mask for cluster 2
    cluster_2_mask = (all_w == 2)
    sigma_2_numerator = np.sum(cluster_2_mask * all_Z**2)  # Sum of Z^2 for cluster 2
    sigma_2_denominator = np.sum(cluster_2_mask)          # Count of samples in cluster 2
    sigma_2_squared = sigma_2_numerator / sigma_2_denominator if sigma_2_denominator > 0 else 0
    sigma_2 = np.sqrt(sigma_2_squared)

    return sigma_1, sigma_2


def M_step(Y, X, all_w, all_Z, updatebeta=True):
    """
    Perform the M-step of the MCEM algorithm, updating beta, sigma, and pi.
    
    Args:
        Y: Observed responses (n x T), binary values {0, 1}
        X: Fixed effects (n x T)
        all_w: Updated cluster assignments (K x n)
        all_Z: Updated random effects (K x n)
        beta1, beta2: Regression coefficients for two clusters
        sigma1, sigma2: Standard deviations for two clusters
        pi1: Prior probability for cluster 1

    Returns:
        beta1, beta2: Updated regression coefficients
        sigma1, sigma2: Updated standard deviations
        pi1: Updated prior probability
    """
    # Update beta values
    beta1, beta2 = updating_beta_c_numba(X, Y, all_w, all_Z, update=updatebeta)
    
    # Update sigma values
    sigma1, sigma2 = compute_sigma(all_w, all_Z)
    
    # Update pi values
    pi1, pi2 = updating_pai_c(all_w)
    
    return beta1, beta2, sigma1, sigma2, pi1


def MCEM(Y, X, K=500, k=100, max_iter=10):
    """
    Perform the MCEM algorithm for a given dataset.
    
    Args:
        Y: Observed responses (n x T), binary values {0, 1}
        X: Fixed effects (n x T)
        Z: Initial random effects (n x 1)
        beta1, beta2: Initial regression coefficients for two clusters
        sigma1, sigma2: Initial standard deviations for two clusters
        pi1: Initial prior probability for cluster 1
        K: Number of iterations for sampling w and Z
        k: Number of burn-in iterations
        max_iter: Maximum number of iterations for the EM algorithm

    Returns:
        beta1, beta2: Final regression coefficients
        sigma1, sigma2: Final standard deviations
        pi1: Final prior probability
    """
    Z = np.random.normal(0, 5, Y.shape[0])  # Initial random effects
    beta1, beta2 = np.random.normal(1, 1, 2)  # Initial regression coefficients
    # beta1, beta2 = 1, 1.5
    sigma1, sigma2 = np.random.uniform(0, 10, 2)  # Initial standard deviations
    # sigma1, sigma2 = 1, 10
    pi1 = np.random.uniform(0.5, 1)  # Initial prior probability
    prev_beta1, prev_beta2, prev_sigma1, prev_sigma2, prev_pi1 = beta1, beta2, sigma1, sigma2, pi1

    for _ in range(max_iter):
        # E-step
        start_time = time.time()
        all_w, all_Z = E_step(Y, X, Z, beta1, beta2, sigma1, sigma2, pi1, K, k)
        print(f"E-step Time: {time.time() - start_time}")

        # M-step
        beta1, beta2, sigma1, sigma2, pi1 = M_step(Y, X, all_w, all_Z, updatebeta=True)
        print(f"Iter: {_}, beta1: {beta1}, beta2: {beta2}, sigma1: {sigma1}, sigma2: {sigma2}, pi1: {pi1}, Time: {time.time() - start_time}")

        # Check for convergence
        converged = True
        for current, prev in zip([beta1, beta2, sigma1, sigma2, pi1], [prev_beta1, prev_beta2, prev_sigma1, prev_sigma2, prev_pi1]):
            if not (0.99 * prev <= current <= 1.01 * prev):  # Check if current value is within ±10% of previous value
                converged = False
                break

        if converged:
            print(f"Converged at iteration {_}")
            break
        prev_beta1, prev_beta2, prev_sigma1, prev_sigma2, prev_pi1 = beta1, beta2, sigma1, sigma2, pi1

    return beta1, beta2, sigma1, sigma2, pi1


def simulation(X, beta1_true, beta2_true, sigma1_true, sigma2_true, pi1_true, K=500, k=100, max_iter=500, ntimes=100):
    results = []   
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.flatten()  
    
    iterations = list(range(1, ntimes + 1))
    
    for i in range(ntimes):
       
        Y = generate_data(X, beta1_true, beta2_true, sigma1_true, sigma2_true, pi1_true)
        beta1_est, beta2_est, sigma1_est, sigma2_est, pi1_est = MCEM(Y, X, K, k, max_iter)
        
        results.append({
            'beta1': beta1_est,
            'beta2': beta2_est,
            'sigma1': sigma1_est,
            'sigma2': sigma2_est,
            'pi1': pi1_est
        })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv('simulation_results.csv', index=False)

        for idx, (param, true_value) in enumerate(zip(['beta1', 'beta2', 'sigma1', 'sigma2', 'pi1'], 
                                                     [beta1_true, beta2_true, sigma1_true, sigma2_true, pi1_true])):
            axes[idx].cla()  # 清除当前轴的内容，避免叠加
            axes[idx].scatter(iterations[:i+1], [result[param] for result in results], label='Estimated Value', color='b')
            axes[idx].axhline(true_value, color='r', linestyle='--', label='True Value')
            axes[idx].set_title(param)
            axes[idx].set_xlabel('Iteration')
            axes[idx].set_ylabel('Value')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(f'Figure/parameter_estimation_{i+1}.png')  # 每次实验后保存为不同的文件
    
    # 返回结果的 DataFrame
    return results_df



if __name__ == '__main__':
    n, T = 500, 20
    X = generate_fixed_X(n, T)
    simulation(X, 1, 1.5, 1, 10, 0.7, K=500, k=100, max_iter=100, ntimes=100)