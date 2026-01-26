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
    
    Args:
        results: Dictionary mapping R to results dict
        R_values: Array of all R values
        thresholds: Dict with 'p0_KS', 'eps_W', 'eps_TV'
        process_name: Name for title (e.g., "α=0.5, r=1, p=3")
        use_KS_stat: If True, plot KS statistic instead of p-value
    
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    R_plot = R_values[::2]  # Plot every other R
    
    for R in R_plot:
        times = results[R]['times']
        
        if use_KS_stat:
            axes[0].plot(times, results[R]['KS_stats'], label=f'R={int(R)}', alpha=0.7)
        else:
            axes[0].plot(times, results[R]['KS_pvals'], label=f'R={int(R)}', alpha=0.7)
        axes[1].plot(times, results[R]['W1'], label=f'R={int(R)}', alpha=0.7)
        axes[2].plot(times, results[R]['TV'], label=f'R={int(R)}', alpha=0.7)
    
    axes[0].axhline(thresholds['p0_KS'], color='red', linestyle='--', linewidth=2, 
                    label=f"threshold={thresholds['p0_KS']}")
    axes[0].set_xlabel('Time t')
    if use_KS_stat:
        axes[0].set_ylabel('KS statistic')
        axes[0].set_title(f'{process_name}: KS statistic vs time')
    else:
        axes[0].set_ylabel('KS p-value')
        axes[0].set_title(f'{process_name}: KS p-value vs time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].axhline(thresholds['eps_W'], color='red', linestyle='--', linewidth=2,
                    label=f"threshold={thresholds['eps_W']}")
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('W₁ distance')
    axes[1].set_title(f'{process_name}: Wasserstein distance vs time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].axhline(thresholds['eps_TV'], color='red', linestyle='--', linewidth=2,
                    label=f"threshold={thresholds['eps_TV']}")
    axes[2].set_xlabel('Time t')
    axes[2].set_ylabel('TV distance')
    axes[2].set_title(f'{process_name}: TV distance vs time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_combined_comparison(all_results, R_values, process_configs, thresholds, use_KS_stat=False):
    """
    Plot combined comparison of all processes for a single R.
    
    Args:
        all_results: Dictionary mapping config_name to results dict
        R_values: Array of R values
        process_configs: List of (alpha, r, p) triplets
        thresholds: Dict with 'p0_KS', 'eps_W', 'eps_TV'
        use_KS_stat: If True, plot KS statistic instead of p-value
    
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Pick middle R value for comparison
    R_display = R_values[len(R_values)//2]
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    for idx, (alpha, r, drift_power) in enumerate(process_configs):
        config_name = f"a{alpha}_r{r}_p{drift_power}"
        color = colors[idx % len(colors)]
        label = f"α={alpha}, r={r}, p={drift_power}"
        
        times = all_results[config_name][R_display]['times']
        
        if use_KS_stat:
            axes[0].plot(times, all_results[config_name][R_display]['KS_stats'], 
                         label=label, color=color, linewidth=2, alpha=0.7)
        else:
            axes[0].plot(times, all_results[config_name][R_display]['KS_pvals'], 
                         label=label, color=color, linewidth=2, alpha=0.7)
        
        axes[1].plot(times, all_results[config_name][R_display]['W1'], 
                     label=label, color=color, linewidth=2, alpha=0.7)
        axes[2].plot(times, all_results[config_name][R_display]['TV'], 
                     label=label, color=color, linewidth=2, alpha=0.7)
    
    # Add threshold lines
    axes[0].axhline(thresholds['p0_KS'], color='black', linestyle='--', linewidth=1, alpha=0.5,
                    label=f"threshold={thresholds['p0_KS']}")
    axes[1].axhline(thresholds['eps_W'], color='black', linestyle='--', linewidth=1, alpha=0.5,
                    label=f"threshold={thresholds['eps_W']}")
    axes[2].axhline(thresholds['eps_TV'], color='black', linestyle='--', linewidth=1, alpha=0.5,
                    label=f"threshold={thresholds['eps_TV']}")
    
    # Labels
    axes[0].set_xlabel('Time t')
    if use_KS_stat:
        axes[0].set_ylabel('KS statistic')
        axes[0].set_title(f'KS statistic comparison (R={int(R_display)})')
    else:
        axes[0].set_ylabel('KS p-value')
        axes[0].set_title(f'KS p-value comparison (R={int(R_display)})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('W₁ distance')
    axes[1].set_title(f'Wasserstein distance comparison (R={int(R_display)})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Time t')
    axes[2].set_ylabel('TV distance')
    axes[2].set_title(f'TV distance comparison (R={int(R_display)})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_asymptotic_scaling(mixing_times, slope_theory, process_name, color):
    """
    Plot asymptotic scaling (mixing time vs log R) for one process.
    
    Args:
        mixing_times: Dict with 'KS', 'W1', 'TV', 'R' arrays
        slope_theory: Not used (kept for compatibility)
        process_name: Name for title
        color: Color for plots
    
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    log_R = np.log(mixing_times['R'])
    fit_KS = np.polyfit(log_R, mixing_times['KS'], 1)
    fit_W1 = np.polyfit(log_R, mixing_times['W1'], 1)
    fit_TV = np.polyfit(log_R, mixing_times['TV'], 1)
    
    log_R_dense = np.linspace(log_R.min(), log_R.max(), 100)
    
    axes[0].plot(log_R, mixing_times['KS'], 'o', color=color, markersize=8, label=process_name)
    axes[0].plot(log_R_dense, fit_KS[0] * log_R_dense + fit_KS[1], '--',
                 color=color, alpha=0.7, label=f'Fitted: slope={fit_KS[0]:.2f}')
    axes[0].set_xlabel('log(R)')
    axes[0].set_ylabel('Mixing time (KS)')
    axes[0].set_title(f'{process_name}: KS mixing time\nFitted slope: {fit_KS[0]:.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(log_R, mixing_times['W1'], 'o', color=color, markersize=8, label=process_name)
    axes[1].plot(log_R_dense, fit_W1[0] * log_R_dense + fit_W1[1], '--',
                 color=color, alpha=0.7, label=f'Fitted: slope={fit_W1[0]:.2f}')
    axes[1].set_xlabel('log(R)')
    axes[1].set_ylabel('Mixing time (W₁)')
    axes[1].set_title(f'{process_name}: Wasserstein mixing time\nFitted slope: {fit_W1[0]:.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(log_R, mixing_times['TV'], 'o', color=color, markersize=8, label=process_name)
    axes[2].plot(log_R_dense, fit_TV[0] * log_R_dense + fit_TV[1], '--',
                 color=color, alpha=0.7, label=f'Fitted: slope={fit_TV[0]:.2f}')
    axes[2].set_xlabel('log(R)')
    axes[2].set_ylabel('Mixing time (TV)')
    axes[2].set_title(f'{process_name}: TV mixing time\nFitted slope: {fit_TV[0]:.2f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_combined_asymptotic(all_mixing_times, process_configs):
    """
    Plot combined asymptotic scaling comparison for all processes.
    
    Args:
        all_mixing_times: Dictionary mapping config_name to mixing_times dict
        process_configs: List of (alpha, r, p) triplets
    
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    markers = ['o', 's', '^', 'v', 'D', 'p']
    
    for idx, (alpha, r, drift_power) in enumerate(process_configs):
        config_name = f"a{alpha}_r{r}_p{drift_power}"
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        label = f"α={alpha}, r={r}, p={drift_power}"
        
        mixing_times = all_mixing_times[config_name]
        log_R = np.log(mixing_times['R'])
        log_R_dense = np.linspace(log_R.min(), log_R.max(), 100)
        
        # Fit slopes
        fit_KS = np.polyfit(log_R, mixing_times['KS'], 1)
        fit_W1 = np.polyfit(log_R, mixing_times['W1'], 1)
        fit_TV = np.polyfit(log_R, mixing_times['TV'], 1)
        
        # KS
        axes[0].plot(log_R, mixing_times['KS'], marker, color=color, markersize=8, 
                     label=f"{label} ({fit_KS[0]:.2f})", alpha=0.7)
        axes[0].plot(log_R_dense, fit_KS[0] * log_R_dense + fit_KS[1], '--', 
                     color=color, alpha=0.5)
        
        # W1
        axes[1].plot(log_R, mixing_times['W1'], marker, color=color, markersize=8, 
                     label=f"{label} ({fit_W1[0]:.2f})", alpha=0.7)
        axes[1].plot(log_R_dense, fit_W1[0] * log_R_dense + fit_W1[1], '--', 
                     color=color, alpha=0.5)
        
        # TV
        axes[2].plot(log_R, mixing_times['TV'], marker, color=color, markersize=8, 
                     label=f"{label} ({fit_TV[0]:.2f})", alpha=0.7)
        axes[2].plot(log_R_dense, fit_TV[0] * log_R_dense + fit_TV[1], '--', 
                     color=color, alpha=0.5)
    
    # Labels
    axes[0].set_xlabel('log(R)')
    axes[0].set_ylabel('Mixing time (KS)')
    axes[0].set_title('KS mixing time comparison\n(slopes in legend)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('log(R)')
    axes[1].set_ylabel('Mixing time (W₁)')
    axes[1].set_title('Wasserstein mixing time comparison\n(slopes in legend)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('log(R)')
    axes[2].set_ylabel('Mixing time (TV)')
    axes[2].set_title('TV mixing time comparison\n(slopes in legend)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig