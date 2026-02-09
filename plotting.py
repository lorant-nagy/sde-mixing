# plotting.py
"""
Plotting functions for visualization of multiple process configurations.
All functions return matplotlib figure objects (don't show or save).
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_cutoff_shapes(results, R_values, thresholds, process_name, use_KS_stat=False):
    """
    Plot cutoff shapes (KS, W1, TV vs time) for multiple R values.
    Returns 3 individual figures.
    
    Args:
        results: Dictionary mapping R to results dict
        R_values: Array of all R values
        thresholds: Dict with 'p0_KS', 'eps_W', 'eps_TV'
        process_name: Name for title (e.g., "α=0.5, r=1, p=3")
        use_KS_stat: If True, plot KS statistic instead of p-value
    
    Returns:
        Dictionary with 'KS', 'W1', 'TV' figure objects
    """
    R_plot = R_values[::2]  # Plot every other R
    
    # KS figure
    fig_ks = plt.figure(figsize=(8, 6), dpi=150)
    ax_ks = fig_ks.add_subplot(111)
    for R in R_plot:
        times = results[R]['times']
        if use_KS_stat:
            ax_ks.plot(times, results[R]['KS_stats'], label=f'R={int(R)}', alpha=0.7)
        else:
            ax_ks.plot(times, results[R]['KS_pvals'], label=f'R={int(R)}', alpha=0.7)
    
    ax_ks.axhline(thresholds['p0_KS'], color='red', linestyle='--', linewidth=2, 
                   label=f"threshold={thresholds['p0_KS']}")
    ax_ks.set_xlabel('Time t')
    if use_KS_stat:
        ax_ks.set_ylabel('KS statistic')
        ax_ks.set_title(f'{process_name}: KS statistic vs time')
    else:
        ax_ks.set_ylabel('KS p-value')
        ax_ks.set_title(f'{process_name}: KS p-value vs time')
    ax_ks.legend()
    ax_ks.grid(True, alpha=0.3)
    fig_ks.tight_layout()
    
    # Wasserstein figure
    fig_w1 = plt.figure(figsize=(8, 6), dpi=150)
    ax_w1 = fig_w1.add_subplot(111)
    for R in R_plot:
        times = results[R]['times']
        ax_w1.plot(times, results[R]['W1'], label=f'R={int(R)}', alpha=0.7)
    
    ax_w1.axhline(thresholds['eps_W'], color='red', linestyle='--', linewidth=2,
                   label=f"threshold={thresholds['eps_W']}")
    ax_w1.set_xlabel('Time t')
    ax_w1.set_ylabel('W₁ distance')
    ax_w1.set_title(f'{process_name}: Wasserstein-1 distance vs time')
    ax_w1.legend()
    ax_w1.grid(True, alpha=0.3)
    fig_w1.tight_layout()
    
    # TV figure
    fig_tv = plt.figure(figsize=(8, 6), dpi=150)
    ax_tv = fig_tv.add_subplot(111)
    for R in R_plot:
        times = results[R]['times']
        ax_tv.plot(times, results[R]['TV'], label=f'R={int(R)}', alpha=0.7)
    
    ax_tv.axhline(thresholds['eps_TV'], color='red', linestyle='--', linewidth=2,
                   label=f"threshold={thresholds['eps_TV']}")
    ax_tv.set_xlabel('Time t')
    ax_tv.set_ylabel('TV distance')
    ax_tv.set_title(f'{process_name}: TV distance vs time')
    ax_tv.legend()
    ax_tv.grid(True, alpha=0.3)
    fig_tv.tight_layout()
    
    return {'KS': fig_ks, 'W1': fig_w1, 'TV': fig_tv}


def plot_combined_comparison(all_results, R_values, process_configs, thresholds, use_KS_stat=False):
    """
    Plot combined comparison of all processes for multiple R values.
    Creates individual plots for each metric and R value.
    
    Args:
        all_results: Dictionary mapping config_name to results dict
        R_values: Array of R values
        process_configs: List of (alpha, r, p) triplets
        thresholds: Dict with 'p0_KS', 'eps_W', 'eps_TV'
        use_KS_stat: If True, plot KS statistic instead of p-value
    
    Returns:
        Dictionary mapping (metric, R) to figure object
    """
    # Select R values at 1/3, 1/2, 3/4, and last position
    n = len(R_values)
    R_indices = [
        n // 3,       # 1/3 position
        n // 2,       # 1/2 position
        3 * n // 4,   # 3/4 position
        n - 1         # Last position
    ]
    R_display_values = [R_values[i] for i in R_indices]
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    figures = {}
    
    for R_display in R_display_values:
        # KS figure
        fig_ks = plt.figure(figsize=(8, 6), dpi=150)
        ax_ks = fig_ks.add_subplot(111)
        
        for idx, (alpha, r, drift_power) in enumerate(process_configs):
            config_name = f"a{alpha}_r{r}_p{drift_power}"
            color = colors[idx % len(colors)]
            label = f"α={alpha}, r={r}, p={drift_power}"
            
            times = all_results[config_name][R_display]['times']
            
            if use_KS_stat:
                ax_ks.plot(times, all_results[config_name][R_display]['KS_stats'], 
                          label=label, color=color, linewidth=2, alpha=0.7)
            else:
                ax_ks.plot(times, all_results[config_name][R_display]['KS_pvals'], 
                          label=label, color=color, linewidth=2, alpha=0.7)
        
        ax_ks.axhline(thresholds['p0_KS'], color='black', linestyle='--', linewidth=1, alpha=0.5,
                     label=f"threshold={thresholds['p0_KS']}")
        ax_ks.set_xlabel('Time t')
        if use_KS_stat:
            ax_ks.set_ylabel('KS statistic')
            ax_ks.set_title(f'KS statistic comparison (R={R_display:.1f})')
        else:
            ax_ks.set_ylabel('KS p-value')
            ax_ks.set_title(f'KS p-value comparison (R={R_display:.1f})')
        ax_ks.legend()
        ax_ks.grid(True, alpha=0.3)
        fig_ks.tight_layout()
        figures[('KS', R_display)] = fig_ks
        
        # Wasserstein figure
        fig_w1 = plt.figure(figsize=(8, 6), dpi=150)
        ax_w1 = fig_w1.add_subplot(111)
        
        for idx, (alpha, r, drift_power) in enumerate(process_configs):
            config_name = f"a{alpha}_r{r}_p{drift_power}"
            color = colors[idx % len(colors)]
            label = f"α={alpha}, r={r}, p={drift_power}"
            
            times = all_results[config_name][R_display]['times']
            ax_w1.plot(times, all_results[config_name][R_display]['W1'], 
                      label=label, color=color, linewidth=2, alpha=0.7)
        
        ax_w1.axhline(thresholds['eps_W'], color='black', linestyle='--', linewidth=1, alpha=0.5,
                     label=f"threshold={thresholds['eps_W']}")
        ax_w1.set_xlabel('Time t')
        ax_w1.set_ylabel('W₁ distance')
        ax_w1.set_title(f'Wasserstein-1 distance comparison (R={R_display:.1f})')
        ax_w1.legend()
        ax_w1.grid(True, alpha=0.3)
        fig_w1.tight_layout()
        figures[('W1', R_display)] = fig_w1
        
        # TV figure
        fig_tv = plt.figure(figsize=(8, 6), dpi=150)
        ax_tv = fig_tv.add_subplot(111)
        
        for idx, (alpha, r, drift_power) in enumerate(process_configs):
            config_name = f"a{alpha}_r{r}_p{drift_power}"
            color = colors[idx % len(colors)]
            label = f"α={alpha}, r={r}, p={drift_power}"
            
            times = all_results[config_name][R_display]['times']
            ax_tv.plot(times, all_results[config_name][R_display]['TV'], 
                      label=label, color=color, linewidth=2, alpha=0.7)
        
        ax_tv.axhline(thresholds['eps_TV'], color='black', linestyle='--', linewidth=1, alpha=0.5,
                     label=f"threshold={thresholds['eps_TV']}")
        ax_tv.set_xlabel('Time t')
        ax_tv.set_ylabel('TV distance')
        ax_tv.set_title(f'TV distance comparison (R={R_display:.1f})')
        ax_tv.legend()
        ax_tv.grid(True, alpha=0.3)
        fig_tv.tight_layout()
        figures[('TV', R_display)] = fig_tv
    
    return figures


def plot_asymptotic_scaling(mixing_times, slope_theory, process_name, color):
    """
    Plot asymptotic scaling (mixing time vs log R) for one process.
    Returns 3 individual figures.
    
    Args:
        mixing_times: Dict with 'KS', 'W1', 'TV', 'R' arrays
        slope_theory: Not used (kept for compatibility)
        process_name: Name for title
        color: Color for plots
    
    Returns:
        Dictionary with 'KS', 'W1', 'TV' figure objects
    """
    log_R = np.log(mixing_times['R'])
    fit_KS = np.polyfit(log_R, mixing_times['KS'], 1)
    fit_W1 = np.polyfit(log_R, mixing_times['W1'], 1)
    fit_TV = np.polyfit(log_R, mixing_times['TV'], 1)
    
    log_R_dense = np.linspace(log_R.min(), log_R.max(), 100)
    
    # KS figure
    fig_ks = plt.figure(figsize=(8, 6), dpi=150)
    ax_ks = fig_ks.add_subplot(111)
    ax_ks.plot(log_R, mixing_times['KS'], 'o', color=color, markersize=8, label=process_name)
    ax_ks.plot(log_R_dense, fit_KS[0] * log_R_dense + fit_KS[1], '--',
                color=color, alpha=0.7, label=f'Fitted: slope={fit_KS[0]:.2f}')
    ax_ks.set_xlabel('log(R)')
    ax_ks.set_ylabel('Mixing time (KS)')
    ax_ks.set_title(f'{process_name}: KS mixing time\nFitted slope: {fit_KS[0]:.2f}')
    ax_ks.legend()
    ax_ks.grid(True, alpha=0.3)
    fig_ks.tight_layout()
    
    # Wasserstein figure
    fig_w1 = plt.figure(figsize=(8, 6), dpi=150)
    ax_w1 = fig_w1.add_subplot(111)
    ax_w1.plot(log_R, mixing_times['W1'], 'o', color=color, markersize=8, label=process_name)
    ax_w1.plot(log_R_dense, fit_W1[0] * log_R_dense + fit_W1[1], '--',
                color=color, alpha=0.7, label=f'Fitted: slope={fit_W1[0]:.2f}')
    ax_w1.set_xlabel('log(R)')
    ax_w1.set_ylabel('Mixing time (W₁)')
    ax_w1.set_title(f'{process_name}: Wasserstein-1 mixing time\nFitted slope: {fit_W1[0]:.2f}')
    ax_w1.legend()
    ax_w1.grid(True, alpha=0.3)
    fig_w1.tight_layout()
    
    # TV figure
    fig_tv = plt.figure(figsize=(8, 6), dpi=150)
    ax_tv = fig_tv.add_subplot(111)
    ax_tv.plot(log_R, mixing_times['TV'], 'o', color=color, markersize=8, label=process_name)
    ax_tv.plot(log_R_dense, fit_TV[0] * log_R_dense + fit_TV[1], '--',
                color=color, alpha=0.7, label=f'Fitted: slope={fit_TV[0]:.2f}')
    ax_tv.set_xlabel('log(R)')
    ax_tv.set_ylabel('Mixing time (TV)')
    ax_tv.set_title(f'{process_name}: TV mixing time\nFitted slope: {fit_TV[0]:.2f}')
    ax_tv.legend()
    ax_tv.grid(True, alpha=0.3)
    fig_tv.tight_layout()
    
    return {'KS': fig_ks, 'W1': fig_w1, 'TV': fig_tv}


def plot_combined_asymptotic(all_mixing_times, process_configs):
    """
    Plot combined asymptotic scaling comparison for all processes.
    Returns 3 individual figures.
    
    Args:
        all_mixing_times: Dictionary mapping config_name to mixing_times dict
        process_configs: List of (alpha, r, p) triplets
    
    Returns:
        Dictionary with 'KS', 'W1', 'TV' figure objects
    """
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    markers = ['o', 's', '^', 'v', 'D', 'p']
    
    # KS figure
    fig_ks = plt.figure(figsize=(10, 7), dpi=150)
    ax_ks = fig_ks.add_subplot(111)
    
    for idx, (alpha, r, drift_power) in enumerate(process_configs):
        config_name = f"a{alpha}_r{r}_p{drift_power}"
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        label = f"α={alpha}, r={r}, p={drift_power}"
        
        mixing_times = all_mixing_times[config_name]
        log_R = np.log(mixing_times['R'])
        log_R_dense = np.linspace(log_R.min(), log_R.max(), 100)
        
        fit_KS = np.polyfit(log_R, mixing_times['KS'], 1)
        ax_ks.plot(log_R, mixing_times['KS'], marker, color=color, markersize=8, 
                  label=f"{label} ({fit_KS[0]:.2f})", alpha=0.7)
        ax_ks.plot(log_R_dense, fit_KS[0] * log_R_dense + fit_KS[1], '--', 
                  color=color, alpha=0.5)
    
    ax_ks.set_xlabel('log(R)')
    ax_ks.set_ylabel('Mixing time (KS)')
    ax_ks.set_title('KS mixing time comparison\n(slopes in legend)')
    ax_ks.legend(fontsize=8)
    ax_ks.grid(True, alpha=0.3)
    fig_ks.tight_layout()
    
    # Wasserstein figure
    fig_w1 = plt.figure(figsize=(10, 7), dpi=150)
    ax_w1 = fig_w1.add_subplot(111)
    
    for idx, (alpha, r, drift_power) in enumerate(process_configs):
        config_name = f"a{alpha}_r{r}_p{drift_power}"
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        label = f"α={alpha}, r={r}, p={drift_power}"
        
        mixing_times = all_mixing_times[config_name]
        log_R = np.log(mixing_times['R'])
        log_R_dense = np.linspace(log_R.min(), log_R.max(), 100)
        
        fit_W1 = np.polyfit(log_R, mixing_times['W1'], 1)
        ax_w1.plot(log_R, mixing_times['W1'], marker, color=color, markersize=8, 
                  label=f"{label} ({fit_W1[0]:.2f})", alpha=0.7)
        ax_w1.plot(log_R_dense, fit_W1[0] * log_R_dense + fit_W1[1], '--', 
                  color=color, alpha=0.5)
    
    ax_w1.set_xlabel('log(R)')
    ax_w1.set_ylabel('Mixing time (W₁)')
    ax_w1.set_title('Wasserstein-1 mixing time comparison\n(slopes in legend)')
    ax_w1.legend(fontsize=8)
    ax_w1.grid(True, alpha=0.3)
    fig_w1.tight_layout()
    
    # TV figure
    fig_tv = plt.figure(figsize=(10, 7), dpi=150)
    ax_tv = fig_tv.add_subplot(111)
    
    for idx, (alpha, r, drift_power) in enumerate(process_configs):
        config_name = f"a{alpha}_r{r}_p{drift_power}"
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        label = f"α={alpha}, r={r}, p={drift_power}"
        
        mixing_times = all_mixing_times[config_name]
        log_R = np.log(mixing_times['R'])
        log_R_dense = np.linspace(log_R.min(), log_R.max(), 100)
        
        fit_TV = np.polyfit(log_R, mixing_times['TV'], 1)
        ax_tv.plot(log_R, mixing_times['TV'], marker, color=color, markersize=8, 
                  label=f"{label} ({fit_TV[0]:.2f})", alpha=0.7)
        ax_tv.plot(log_R_dense, fit_TV[0] * log_R_dense + fit_TV[1], '--', 
                  color=color, alpha=0.5)
    
    ax_tv.set_xlabel('log(R)')
    ax_tv.set_ylabel('Mixing time (TV)')
    ax_tv.set_title('TV mixing time comparison\n(slopes in legend)')
    ax_tv.legend(fontsize=8)
    ax_tv.grid(True, alpha=0.3)
    fig_tv.tight_layout()
    
    return {'KS': fig_ks, 'W1': fig_w1, 'TV': fig_tv}

def plot_initial_distribution_heatmap(R_values, samples_dict, process_name, delta):
    """
    Create heatmap of initial distributions across R values.
    Returns figure object (no disk writes).
    
    Args:
        R_values: Array of R values
        samples_dict: {R: samples_array} - initial samples for each R
        process_name: String for title (e.g., "α=0.5, r=1, p=3")
        delta: Parameter controlling far cluster width in initial distribution
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
    
    # Determine global bin range
    R_max = R_values.max()
    x_max = (1 + 2*delta) * R_max
    bins = np.linspace(-x_max, x_max, 200)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Build probability matrix
    n_R = len(R_values)
    n_bins = len(bins) - 1
    P = np.zeros((n_R, n_bins))
    
    for i, R in enumerate(R_values):
        counts, _ = np.histogram(samples_dict[R], bins=bins)
        P[i, :] = counts / counts.sum()  # Normalize to probabilities
    
    # Plot heatmap
    im = ax.imshow(P, aspect='auto', origin='lower', 
                   extent=[bin_centers[0], bin_centers[-1], 0, n_R-1],
                   cmap='viridis', interpolation='nearest')
    
    # Y-axis: show actual R values
    ax.set_yticks(range(n_R))
    ax.set_yticklabels([f'{int(R)}' for R in R_values])
    
    ax.set_xlabel('x (state)')
    ax.set_ylabel('R (initial radius)')
    ax.set_title(f'Initial Distribution Heatmap: {process_name}')
    plt.colorbar(im, ax=ax, label='Probability density')
    
    plt.tight_layout()
    return fig