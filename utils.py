# utils.py
"""
Utility functions for simulation and metric computation.
"""
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.special import gamma


def simulate_OU(R, mu, sigma, dt, N, M, every_k):
    """
    OU process: dX = -mu * X * dt + sigma * dW
    
    Args:
        R: Initial radius
        mu: Drift parameter
        sigma: Diffusion parameter
        dt: Time step
        N: Total number of steps
        M: Number of trajectories
        every_k: Save every k-th step
    
    Returns:
        times: Array of time points
        samples: Array of shape (n_save, M) with samples
    """
    X = np.full(M, R, dtype=np.float64)
    
    n_save = N // every_k + 1
    times = np.zeros(n_save)
    samples = np.zeros((n_save, M))
    
    samples[0] = X
    times[0] = 0.0
    
    save_idx = 1
    for n in range(1, N + 1):
        Z = np.random.randn(M)
        X = X - mu * X * dt + sigma * np.sqrt(dt) * Z
        
        if n % every_k == 0:
            times[save_idx] = n * dt
            samples[save_idx] = X
            save_idx += 1
    
    # Check for NaN values
    assert not np.any(np.isnan(times)), "NaN detected in OU times"
    assert not np.any(np.isnan(samples)), "NaN detected in OU samples"
    
    return times, samples


def simulate_superlinear_tamed(R, mu, sigma, power, dt, N, M, every_k):
    """
    Tamed Euler-Maruyama for superlinear Langevin: 
    dX = -mu * |X|^(p-1) * X * dt + sigma * dW
    
    Taming prevents numerical explosion by dividing drift by (1 + dt * |drift|)
    
    Args:
        R: Initial radius
        mu: Drift parameter
        sigma: Diffusion parameter
        power: Power in drift (p=3 for cubic)
        dt: Time step
        N: Total number of steps
        M: Number of trajectories
        every_k: Save every k-th step
    
    Returns:
        times: Array of time points
        samples: Array of shape (n_save, M) with samples
    """
    X = np.full(M, R, dtype=np.float64)
    
    n_save = N // every_k + 1
    times = np.zeros(n_save)
    samples = np.zeros((n_save, M))
    
    samples[0] = X
    times[0] = 0.0
    
    save_idx = 1
    for n in range(1, N + 1):
        Z = np.random.randn(M)
        
        # Compute raw drift
        drift_raw = -mu * np.abs(X)**(power - 1) * X
        
        # Taming: divide by (1 + dt * |drift|)
        drift_norm = np.abs(drift_raw)
        drift = drift_raw / (1.0 + dt * drift_norm)
        
        X = X + drift * dt + sigma * np.sqrt(dt) * Z
        
        if n % every_k == 0:
            times[save_idx] = n * dt
            samples[save_idx] = X
            save_idx += 1
    
    # Check for NaN values
    assert not np.any(np.isnan(times)), "NaN detected in superlinear times"
    assert not np.any(np.isnan(samples)), "NaN detected in superlinear samples"
    
    return times, samples


def sample_stationary_superlinear_exact(mu, sigma, power, n):
    """
    Exact sampling from the stationary distribution of superlinear Langevin.
    
    For dX = -mu * |X|^(p-1) * X * dt + sigma * dW,
    the stationary density is proportional to exp(-alpha * |x|^k)
    where k = p + 1 and alpha = 2*mu / (k * sigma^2).
    
    This uses the Gamma distribution transformation:
    U = alpha * |X|^k ~ Gamma(1/k, 1)
    
    Args:
        mu: Drift parameter
        sigma: Diffusion parameter
        power: Power in drift (p)
        n: Number of samples
    
    Returns:
        Array of n samples from stationary distribution
    """
    k = power + 1
    alpha = 2 * mu / (k * sigma**2)
    
    # Sample U ~ Gamma(1/k, 1)
    u = np.random.gamma(shape=1.0/k, scale=1.0, size=n)
    
    # Transform to get |X|
    r = (u / alpha)**(1.0 / k)
    
    # Random signs
    signs = np.where(np.random.rand(n) < 0.5, -1.0, 1.0)
    
    return signs * r


def get_stationary_std_superlinear(mu, sigma, power):
    """
    Compute the theoretical standard deviation of the stationary distribution.
    
    For the superlinear Langevin with density proportional to exp(-alpha * |x|^k),
    the variance is: 2 * Gamma(3/k) / (alpha^(2/k) * Gamma(1/k))
    
    Args:
        mu: Drift parameter
        sigma: Diffusion parameter
        power: Power in drift (p)
    
    Returns:
        Standard deviation of stationary distribution
    """
    k = power + 1
    alpha = 2 * mu / (k * sigma**2)
    
    variance = 2 * gamma(3.0/k) / (alpha**(2.0/k) * gamma(1.0/k))
    return np.sqrt(variance)


def compute_metrics(samples, stationary_ref, stationary_std, R_max):
    """
    Compute KS p-value, KS statistic, Wasserstein distance, and TV distance for all time points.
    Uses process-specific stationary reference.
    
    Args:
        samples: Array of shape (n_times, M) with samples at each time
        stationary_ref: Stationary reference samples
        stationary_std: Standard deviation of stationary distribution
        R_max: Maximum R value (for bin range)
    
    Returns:
        KS_pvals: KS test p-values
        KS_stats: KS test statistics (distance)
        W1_dists: Wasserstein distances
        TV_dists: Total variation distances
    """
    # Check for NaN in inputs
    assert not np.any(np.isnan(samples)), "NaN detected in samples input to compute_metrics"
    assert not np.any(np.isnan(stationary_ref)), "NaN detected in stationary_ref input to compute_metrics"
    
    n_times = samples.shape[0]
    KS_pvals = np.zeros(n_times)
    KS_stats = np.zeros(n_times)
    W1_dists = np.zeros(n_times)
    TV_dists = np.zeros(n_times)
    
    # Use bins that cover both stationary distribution and transient states
    bin_range = max(5 * stationary_std, 1.2 * R_max)
    bins = np.linspace(-bin_range, bin_range, 100)
    
    # Precompute stationary histogram (as probabilities)
    counts_stat, _ = np.histogram(stationary_ref, bins=bins, density=False)
    p_stat = counts_stat / counts_stat.sum()
    
    for i in range(n_times):
        X_t = samples[i]
        
        # KS test (two-sample test against stationary reference)
        ks_stat, p_val = ks_2samp(X_t, stationary_ref)
        KS_pvals[i] = p_val
        KS_stats[i] = ks_stat
        
        # Wasserstein distance
        W1_dists[i] = wasserstein_distance(X_t, stationary_ref)
        
        # TV distance (fixed to handle empty bins)
        counts_samp, _ = np.histogram(X_t, bins=bins, density=False)
        if counts_samp.sum() == 0:
            # All samples outside bin range - maximal disagreement
            TV_dists[i] = 1.0
        else:
            p_samp = counts_samp / counts_samp.sum()
            TV_dists[i] = 0.5 * np.abs(p_samp - p_stat).sum()
    
    # Check for NaN in outputs
    assert not np.any(np.isnan(KS_pvals)), "NaN detected in KS p-values"
    assert not np.any(np.isnan(KS_stats)), "NaN detected in KS statistics"
    assert not np.any(np.isnan(W1_dists)), "NaN detected in Wasserstein distances"
    assert not np.any(np.isnan(TV_dists)), "NaN detected in TV distances"
    
    return KS_pvals, KS_stats, W1_dists, TV_dists


def extract_mixing_times(results, R_values, p0_KS, eps_W, eps_TV, T_max, use_KS_stat=False):
    """
    Extract mixing times for each R value based on thresholds.
    
    Args:
        results: Dictionary mapping R to results dict with 'times', 'KS_pvals', 'KS_stats', 'W1', 'TV'
        R_values: Array of R values
        p0_KS: KS threshold (p-value if use_KS_stat=False, statistic if True)
        eps_W: Wasserstein distance threshold
        eps_TV: TV distance threshold
        T_max: Maximum time (used if threshold never crossed)
        use_KS_stat: If True, use KS statistic instead of p-value
    
    Returns:
        Dictionary with 'KS', 'W1', 'TV', 'R' arrays
    """
    mixing_times = {
        'KS': [],
        'W1': [],
        'TV': [],
        'R': []
    }
    
    for R in R_values:
        times = results[R]['times']
        
        # KS mixing time
        if use_KS_stat:
            # Use KS statistic (want it to go below threshold)
            idx_KS = np.where(results[R]['KS_stats'] <= p0_KS)[0]
        else:
            # Use KS p-value (want it to go above threshold)
            idx_KS = np.where(results[R]['KS_pvals'] >= p0_KS)[0]
        t_KS = times[idx_KS[0]] if len(idx_KS) > 0 else T_max
        
        idx_W = np.where(results[R]['W1'] <= eps_W)[0]
        t_W = times[idx_W[0]] if len(idx_W) > 0 else T_max
        
        idx_TV = np.where(results[R]['TV'] <= eps_TV)[0]
        t_TV = times[idx_TV[0]] if len(idx_TV) > 0 else T_max
        
        mixing_times['KS'].append(t_KS)
        mixing_times['W1'].append(t_W)
        mixing_times['TV'].append(t_TV)
        mixing_times['R'].append(R)
    
    mixing_times['KS'] = np.array(mixing_times['KS'])
    mixing_times['W1'] = np.array(mixing_times['W1'])
    mixing_times['TV'] = np.array(mixing_times['TV'])
    mixing_times['R'] = np.array(mixing_times['R'])
    
    # Check for NaN in mixing times
    assert not np.any(np.isnan(mixing_times['KS'])), "NaN detected in KS mixing times"
    assert not np.any(np.isnan(mixing_times['W1'])), "NaN detected in W1 mixing times"
    assert not np.any(np.isnan(mixing_times['TV'])), "NaN detected in TV mixing times"
    
    return mixing_times