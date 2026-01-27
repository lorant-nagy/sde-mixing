# main.py
"""
Main script to run Sabanis/TUSLA taming comparison across different (α, r, p) configurations.
Logs all results to WandB.
"""
import numpy as np
import os
import wandb
import matplotlib.pyplot as plt

from config import PARAMS
from utils import (
    simulate_tusla,
    sample_initial_bimodal,
    sample_stationary_superlinear_exact,
    get_stationary_std_superlinear,
    compute_metrics,
    extract_mixing_times
)
from plotting import (
    plot_cutoff_shapes,
    plot_combined_comparison,
    plot_asymptotic_scaling,
    plot_combined_asymptotic,
    plot_initial_distribution_heatmap
)


# Initial distribution parameters (bimodal setup from paper)
DELTA = 0.05    # Far cluster relative position/width
EPS = 0.01      # Paper parameter (not used in sampler, kept for reference)
B_RHO = 0.10    # Probability mass in far cluster (10%)
SYMMETRIC = True  # Mirror far cluster to both sides (±x0)


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # WandB environment is configured via docker-compose.yml
    # All files go to /tmp (ephemeral, cleaned up with container)
    
    # Initialize WandB
    wandb.init(
        project=PARAMS['wandb_project'],
        entity=PARAMS['wandb_entity'],
        config=PARAMS
    )
    
    # Extract parameters
    mu = PARAMS['mu']
    sigma = PARAMS['sigma']
    process_configs = PARAMS['process_configs']
    T_max = PARAMS['T_max']
    N = PARAMS['N']
    M = PARAMS['M']
    every_k = PARAMS['every_k']
    R_values = PARAMS['R_values']
    R_max = R_values.max()
    dt = T_max / N
    
    # TUSLA regularization (typically small)
    eta = 0.01
    
    thresholds = {
        'p0_KS': PARAMS['p0_KS'],
        'eps_W': PARAMS['eps_W'],
        'eps_TV': PARAMS['eps_TV']
    }
    
    # Option to use KS statistic instead of p-value (recommended)
    use_KS_stat = False  # Set to True to use KS statistic
    
    print("=" * 80)
    print("PROCESS CONFIGURATIONS")
    print("=" * 80)
    for alpha, r, drift_power in process_configs:
        print(f"  (α={alpha}, r={r}, p={drift_power}): drift=-μ|X|^{drift_power-1}X, taming=1+dt^{alpha}|X|^{2*r}")
    
    print("\n" + "=" * 80)
    print("GENERATING STATIONARY REFERENCES")
    print("=" * 80)
    
    # Generate stationary reference for each process
    stationary_refs = {}
    stationary_stds = {}
    
    for alpha, r, drift_power in process_configs:
        config_name = f"a{alpha}_r{r}_p{drift_power}"
        print(f"\nGenerating stationary reference for (α={alpha}, r={r}, p={drift_power})...")
        
        stationary_ref = sample_stationary_superlinear_exact(
            mu=mu,
            sigma=sigma,
            power=drift_power,
            n=PARAMS['M_stat']
        )
        stationary_std = stationary_ref.std()
        stationary_std_theory = get_stationary_std_superlinear(mu, sigma, drift_power)
        
        stationary_refs[config_name] = stationary_ref
        stationary_stds[config_name] = stationary_std
        
        print(f"  Stationary: mean={stationary_ref.mean():.6f}, std={stationary_std:.6f}")
        print(f"  Theoretical std: {stationary_std_theory:.6f}")
        
        # Log stationary statistics
        wandb.log({
            f"stationary/{config_name}_mean": stationary_ref.mean(),
            f"stationary/{config_name}_std": stationary_std,
            f"stationary/{config_name}_std_theory": stationary_std_theory
        })
    
    print("\n" + "=" * 80)
    print("SIMULATING PROCESSES FOR ALL R")
    print("=" * 80)
    
    # Store results for each process configuration
    all_results = {}
    
    for alpha, r, drift_power in process_configs:
        config_name = f"a{alpha}_r{r}_p{drift_power}"
        print(f"\n{'='*80}")
        print(f"PROCESS: α={alpha}, r={r}, p={drift_power}")
        print(f"{'='*80}")
        
        # Generate bimodal initial samples for all R
        print(f"\nGenerating bimodal initial distribution for all R...")
        samples0_by_R = {
            R: sample_initial_bimodal(R, M, EPS, DELTA, B_RHO, SYMMETRIC)
            for R in R_values
        }
        
        # Create and log initial distribution heatmap
        fig = plot_initial_distribution_heatmap(R_values, samples0_by_R, config_name)
        wandb.log({f"plots/{config_name}_initial_heatmap": wandb.Image(fig)})
        plt.close(fig)
        print(f"  Initial distribution heatmap logged to WandB")
        
        results = {}
        stationary_ref = stationary_refs[config_name]
        stationary_std = stationary_stds[config_name]
        
        for R in R_values:
            print(f"\nSimulating R = {R}")
            
            times, samples = simulate_tusla(
                R=R,
                mu=mu,
                sigma=sigma,
                drift_power=drift_power,
                r=r,
                alpha=alpha,
                eta=eta,
                dt=dt,
                N=N,
                M=M,
                every_k=every_k,
                X0=samples0_by_R[R]  # Use bimodal initial distribution
            )
            
            # Compute metrics
            KS_pvals, KS_stats, W1_dists, TV_dists = compute_metrics(
                samples=samples,
                stationary_ref=stationary_ref,
                stationary_std=stationary_std,
                R_max=R_max
            )
            
            results[R] = {
                'times': times,
                'samples': samples,
                'KS_pvals': KS_pvals,
                'KS_stats': KS_stats,
                'W1': W1_dists,
                'TV': TV_dists
            }
            
            print(f"  Final KS p-value: {KS_pvals[-1]:.6f}, KS stat: {KS_stats[-1]:.6f}")
            print(f"  Final W1: {W1_dists[-1]:.6f}, TV: {TV_dists[-1]:.6f}")
        
        all_results[config_name] = results
    
    print("\n" + "=" * 80)
    print("EXTRACTING MIXING TIMES")
    print("=" * 80)
    
    # Extract mixing times for each process
    all_mixing_times = {}
    
    for alpha, r, drift_power in process_configs:
        config_name = f"a{alpha}_r{r}_p{drift_power}"
        print(f"\nExtracting mixing times for (α={alpha}, r={r}, p={drift_power})...")
        
        mixing_times = extract_mixing_times(
            results=all_results[config_name],
            R_values=R_values,
            p0_KS=thresholds['p0_KS'],
            eps_W=thresholds['eps_W'],
            eps_TV=thresholds['eps_TV'],
            T_max=T_max,
            use_KS_stat=use_KS_stat
        )
        
        all_mixing_times[config_name] = mixing_times
        
        # Fit log(R) vs mixing time slopes
        log_R = np.log(R_values)
        
        fit_KS = np.polyfit(log_R, mixing_times['KS'], 1)
        fit_W = np.polyfit(log_R, mixing_times['W1'], 1)
        fit_TV = np.polyfit(log_R, mixing_times['TV'], 1)
        
        slope_KS = fit_KS[0]
        slope_W = fit_W[0]
        slope_TV = fit_TV[0]
        
        print(f"  Empirically fitted slopes:")
        print(f"    KS: {slope_KS:.4f}")
        print(f"    W₁: {slope_W:.4f}")
        print(f"    TV: {slope_TV:.4f}")
        
        # Log slopes
        wandb.log({
            f"slopes/{config_name}_KS_fitted": slope_KS,
            f"slopes/{config_name}_W1_fitted": slope_W,
            f"slopes/{config_name}_TV_fitted": slope_TV,
        })
    
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    # Generate plots for each process configuration
    for alpha, r, drift_power in process_configs:
        config_name = f"a{alpha}_r{r}_p{drift_power}"
        print(f"\nGenerating plots for (α={alpha}, r={r}, p={drift_power})...")
        
        results = all_results[config_name]
        mixing_times = all_mixing_times[config_name]
        
        # Cutoff shapes plot
        fig = plot_cutoff_shapes(
            results=results,
            R_values=R_values,
            process_name=f"α={alpha}, r={r}, p={drift_power}",
            thresholds=thresholds,
            use_KS_stat=use_KS_stat
        )
        wandb.log({f"plots/{config_name}_cutoff_shapes": wandb.Image(fig)})
        plt.close(fig)
        
        # Asymptotic scaling plot
        fig = plot_asymptotic_scaling(
            mixing_times=mixing_times,
            slope_theory=None,  # Not used in plotting
            process_name=f"α={alpha}, r={r}, p={drift_power}",
            color='blue'  # Will cycle through colors in combined plot
        )
        wandb.log({f"plots/{config_name}_asymptotic_scaling": wandb.Image(fig)})
        plt.close(fig)
    
    # Combined comparison plots if we have multiple processes
    if len(process_configs) > 1:
        print("\nGenerating combined comparison plots...")
        
        # Combined cutoff comparison
        fig = plot_combined_comparison(
            all_results=all_results,
            R_values=R_values,
            process_configs=process_configs,
            thresholds=thresholds,
            use_KS_stat=use_KS_stat
        )
        wandb.log({"plots/combined_cutoff_comparison": wandb.Image(fig)})
        plt.close(fig)
        
        # Combined asymptotic comparison
        fig = plot_combined_asymptotic(
            all_mixing_times=all_mixing_times,
            process_configs=process_configs
        )
        wandb.log({"plots/combined_asymptotic_comparison": wandb.Image(fig)})
        plt.close(fig)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    wandb.finish()


if __name__ == '__main__':
    main()