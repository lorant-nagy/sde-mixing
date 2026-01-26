# plotting.py
"""
Plotting functions for visualization.
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
        process_name: Name for title (e.g., "OU Process")
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


def plot_combined_comparison(results_OU, results_super, R_display, thresholds, drift_power, use_KS_stat=False):
    """
    Plot combined comparison of OU vs Superlinear for a single R.
    
    Args:
        results_OU: Dictionary with OU results
        results_super: Dictionary with superlinear results
        R_display: R value to display
        thresholds: Dict with 'p0_KS', 'eps_W', 'eps_TV'
        drift_power: Power parameter for superlinear drift
        use_KS_stat: If True, plot KS statistic instead of p-value
    
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    times_OU = results_OU[R_display]['times']
    times_super = results_super[R_display]['times']
    
    if use_KS_stat:
        axes[0].plot(times_OU, results_OU[R_display]['KS_stats'], 
                     label='OU', color='blue', linewidth=2)
        axes[0].plot(times_super, results_super[R_display]['KS_stats'], 
                     label=f'Superlinear (p={drift_power}, tamed)', color='orange', linewidth=2)
        axes[0].set_ylabel('KS statistic')
        axes[0].set_title(f'KS statistic vs time (R={int(R_display)})')
    else:
        axes[0].plot(times_OU, results_OU[R_display]['KS_pvals'], 
                     label='OU', color='blue', linewidth=2)
        axes[0].plot(times_super, results_super[R_display]['KS_pvals'], 
                     label=f'Superlinear (p={drift_power}, tamed)', color='orange', linewidth=2)
        axes[0].set_ylabel('KS p-value')
        axes[0].set_title(f'KS p-value vs time (R={int(R_display)})')
    
    axes[0].axhline(thresholds['p0_KS'], color='red', linestyle='--', linewidth=1, alpha=0.5,
                    label=f"threshold={thresholds['p0_KS']}")
    axes[0].set_xlabel('Time t')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(times_OU, results_OU[R_display]['W1'], 
                 label='OU', color='blue', linewidth=2)
    axes[1].plot(times_super, results_super[R_display]['W1'], 
                 label=f'Superlinear (p={drift_power}, tamed)', color='orange', linewidth=2)
    axes[1].axhline(thresholds['eps_W'], color='red', linestyle='--', linewidth=1, alpha=0.5,
                    label=f"threshold={thresholds['eps_W']}")
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('W₁ distance')
    axes[1].set_title(f'Wasserstein distance vs time (R={int(R_display)})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(times_OU, results_OU[R_display]['TV'], 
                 label='OU', color='blue', linewidth=2)
    axes[2].plot(times_super, results_super[R_display]['TV'], 
                 label=f'Superlinear (p={drift_power}, tamed)', color='orange', linewidth=2)
    axes[2].axhline(thresholds['eps_TV'], color='red', linestyle='--', linewidth=1, alpha=0.5,
                    label=f"threshold={thresholds['eps_TV']}")
    axes[2].set_xlabel('Time t')
    axes[2].set_ylabel('TV distance')
    axes[2].set_title(f'TV distance vs time (R={int(R_display)})')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_asymptotic_scaling(mixing_times, slope_theory, process_name, color):
    """
    Plot asymptotic scaling (mixing time vs log R) for one process.
    
    Args:
        mixing_times: Dict with 'KS', 'W1', 'TV', 'R' arrays
        slope_theory: Theoretical slope (1/mu for OU, None for superlinear)
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
    if slope_theory is not None:
        axes[0].plot(log_R_dense, slope_theory * log_R_dense + fit_KS[1], '--', 
                     color=color, label=f'Theory: slope={slope_theory:.2f}')
    else:
        axes[0].plot(log_R_dense, fit_KS[0] * log_R_dense + fit_KS[1], '--',
                     color=color, label=f'Fit slope: {fit_KS[0]:.2f}')
    axes[0].set_xlabel('log(R)')
    axes[0].set_ylabel('Mixing time (KS)')
    axes[0].set_title(f'{process_name}: KS mixing time\nFit slope: {fit_KS[0]:.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(log_R, mixing_times['W1'], 'o', color=color, markersize=8, label=process_name)
    if slope_theory is not None:
        axes[1].plot(log_R_dense, slope_theory * log_R_dense + fit_W1[1], '--',
                     color=color, label=f'Theory: slope={slope_theory:.2f}')
    else:
        axes[1].plot(log_R_dense, fit_W1[0] * log_R_dense + fit_W1[1], '--',
                     color=color, label=f'Fit slope: {fit_W1[0]:.2f}')
    axes[1].set_xlabel('log(R)')
    axes[1].set_ylabel('Mixing time (W₁)')
    axes[1].set_title(f'{process_name}: Wasserstein mixing time\nFit slope: {fit_W1[0]:.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(log_R, mixing_times['TV'], 'o', color=color, markersize=8, label=process_name)
    if slope_theory is not None:
        axes[2].plot(log_R_dense, slope_theory * log_R_dense + fit_TV[1], '--',
                     color=color, label=f'Theory: slope={slope_theory:.2f}')
    else:
        axes[2].plot(log_R_dense, fit_TV[0] * log_R_dense + fit_TV[1], '--',
                     color=color, label=f'Fit slope: {fit_TV[0]:.2f}')
    axes[2].set_xlabel('log(R)')
    axes[2].set_ylabel('Mixing time (TV)')
    axes[2].set_title(f'{process_name}: TV mixing time\nFit slope: {fit_TV[0]:.2f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_combined_asymptotic(mixing_OU, mixing_super, slope_theory, drift_power):
    """
    Plot combined asymptotic scaling comparison.
    
    Args:
        mixing_OU: Mixing times dict for OU
        mixing_super: Mixing times dict for superlinear
        slope_theory: Theoretical slope for OU
        drift_power: Power parameter for superlinear
    
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    log_R = np.log(mixing_OU['R'])
    log_R_super = np.log(mixing_super['R'])
    log_R_dense = np.linspace(log_R.min(), log_R.max(), 100)
    
    fit_KS = np.polyfit(log_R, mixing_OU['KS'], 1)
    fit_W1 = np.polyfit(log_R, mixing_OU['W1'], 1)
    fit_TV = np.polyfit(log_R, mixing_OU['TV'], 1)
    
    fit_KS_super = np.polyfit(log_R_super, mixing_super['KS'], 1)
    fit_W1_super = np.polyfit(log_R_super, mixing_super['W1'], 1)
    fit_TV_super = np.polyfit(log_R_super, mixing_super['TV'], 1)
    
    # KS
    axes[0].plot(log_R, mixing_OU['KS'], 'o', color='blue', markersize=8, label='OU')
    axes[0].plot(log_R_dense, slope_theory * log_R_dense + fit_KS[1], '--', color='blue')
    axes[0].plot(log_R_super, mixing_super['KS'], 's', color='orange', markersize=8, 
                 label=f'Superlinear (p={drift_power}, tamed)')
    axes[0].plot(log_R_dense, fit_KS_super[0] * log_R_dense + fit_KS_super[1], '--', color='orange')
    axes[0].set_xlabel('log(R)')
    axes[0].set_ylabel('Mixing time (KS)')
    axes[0].set_title(f'KS mixing time comparison\nOU: {fit_KS[0]:.2f}, Super: {fit_KS_super[0]:.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Wasserstein
    axes[1].plot(log_R, mixing_OU['W1'], 'o', color='blue', markersize=8, label='OU')
    axes[1].plot(log_R_dense, slope_theory * log_R_dense + fit_W1[1], '--', color='blue')
    axes[1].plot(log_R_super, mixing_super['W1'], 's', color='orange', markersize=8, 
                 label=f'Superlinear (p={drift_power}, tamed)')
    axes[1].plot(log_R_dense, fit_W1_super[0] * log_R_dense + fit_W1_super[1], '--', color='orange')
    axes[1].set_xlabel('log(R)')
    axes[1].set_ylabel('Mixing time (W₁)')
    axes[1].set_title(f'Wasserstein mixing time comparison\nOU: {fit_W1[0]:.2f}, Super: {fit_W1_super[0]:.2f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # TV
    axes[2].plot(log_R, mixing_OU['TV'], 'o', color='blue', markersize=8, label='OU')
    axes[2].plot(log_R_dense, slope_theory * log_R_dense + fit_TV[1], '--', color='blue')
    axes[2].plot(log_R_super, mixing_super['TV'], 's', color='orange', markersize=8, 
                 label=f'Superlinear (p={drift_power}, tamed)')
    axes[2].plot(log_R_dense, fit_TV_super[0] * log_R_dense + fit_TV_super[1], '--', color='orange')
    axes[2].set_xlabel('log(R)')
    axes[2].set_ylabel('Mixing time (TV)')
    axes[2].set_title(f'TV mixing time comparison\nOU: {fit_TV[0]:.2f}, Super: {fit_TV_super[0]:.2f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig