# main.py
"""
Main script to run OU vs Superlinear Langevin comparison.
Logs all results to WandB.
"""
import numpy as np
import wandb
import matplotlib.pyplot as plt

from config import PARAMS
from utils import (
    simulate_OU,
    simulate_superlinear_tamed,
    sample_stationary_superlinear_exact,
    get_stationary_std_superlinear,
    compute_metrics,
    extract_mixing_times
)
from plotting import (
    plot_cutoff_shapes,
    plot_combined_comparison,
    plot_asymptotic_scaling,
    plot_combined_asymptotic
)


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize WandB
    wandb.init(
        project=PARAMS['wandb_project'],
        entity=PARAMS['wandb_entity'],
        config=PARAMS
    )
    
    # Extract parameters
    mu = PARAMS['mu']
    sigma = PARAMS['sigma']
    drift_power = PARAMS['drift_power']
    T_max = PARAMS['T_max']
    N = PARAMS['N']
    M = PARAMS['M']
    every_k = PARAMS['every_k']
    R_values = PARAMS['R_values']
    R_max = R_values.max()
    dt = T_max / N
    
    thresholds = {
        'p0_KS': PARAMS['p0_KS'],
        'eps_W': PARAMS['eps_W'],
        'eps_TV': PARAMS['eps_TV']
    }
    
    # Option to use KS statistic instead of p-value (recommended)
    use_KS_stat = False  # Set to True to use KS statistic
    
    print("=" * 80)
    print("GENERATING STATIONARY REFERENCES")
    print("=" * 80)
    
    # OU stationary reference (Gaussian)
    print("\nGenerating OU stationary reference (Gaussian)...")
    stationary_std_OU = sigma / np.sqrt(2 * mu)
    stationary_ref_OU = np.random.randn(PARAMS['M_stat']) * stationary_std_OU
    assert not np.any(np.isnan(stationary_ref_OU)), "NaN detected in OU stationary reference"
    
    print(f"  OU stationary: mean={stationary_ref_OU.mean():.6f}, std={stationary_ref_OU.std():.6f}")
    print(f"  OU theoretical std: {stationary_std_OU:.6f}")
    
    # Superlinear stationary reference (exact sampling via Gamma distribution)
    print("\nGenerating superlinear stationary reference (exact sampling)...")
    stationary_ref_super = sample_stationary_superlinear_exact(
        mu=mu,
        sigma=sigma,
        power=drift_power,
        n=PARAMS['M_stat']
    )
    stationary_std_super = stationary_ref_super.std()
    stationary_std_super_theory = get_stationary_std_superlinear(mu, sigma, drift_power)
    
    print(f"  Superlinear stationary: mean={stationary_ref_super.mean():.6f}, std={stationary_ref_super.std():.6f}")
    print(f"  Superlinear theoretical std: {stationary_std_super_theory:.6f}")
    
    # Log stationary statistics
    wandb.log({
        "stationary/OU_mean": stationary_ref_OU.mean(),
        "stationary/OU_std": stationary_ref_OU.std(),
        "stationary/OU_std_theory": stationary_std_OU,
        "stationary/super_mean": stationary_ref_super.mean(),
        "stationary/super_std": stationary_ref_super.std(),
        "stationary/super_std_theory": stationary_std_super_theory
    })
    
    print("\n" + "=" * 80)
    print("SIMULATING PROCESSES FOR ALL R")
    print("=" * 80)
    
    results_OU = {}
    results_super = {}
    
    for R in R_values:
        print(f"\nSimulating R = {R}")
        
        # OU process
        print("  Running OU process...")
        times_OU, samples_OU = simulate_OU(R, mu, sigma, dt, N, M, every_k)
        KS_pvals_OU, KS_stats_OU, W1_OU, TV_OU = compute_metrics(
            samples_OU, stationary_ref_OU, stationary_std_OU, R_max
        )
        results_OU[R] = {
            'times': times_OU,
            'KS_pvals': KS_pvals_OU,
            'KS_stats': KS_stats_OU,
            'W1': W1_OU,
            'TV': TV_OU
        }
        
        # Superlinear process
        print("  Running superlinear process...")
        times_super, samples_super = simulate_superlinear_tamed(R, mu, sigma, drift_power, dt, N, M, every_k)
        KS_pvals_super, KS_stats_super, W1_super, TV_super = compute_metrics(
            samples_super, stationary_ref_super, stationary_std_super, R_max
        )
        results_super[R] = {
            'times': times_super,
            'KS_pvals': KS_pvals_super,
            'KS_stats': KS_stats_super,
            'W1': W1_super,
            'TV': TV_super
        }
    
    print("\n" + "=" * 80)
    print("EXTRACTING MIXING TIMES")
    print("=" * 80)
    
    mixing_OU = extract_mixing_times(
        results_OU, R_values, thresholds['p0_KS'], 
        thresholds['eps_W'], thresholds['eps_TV'], T_max, use_KS_stat
    )
    mixing_super = extract_mixing_times(
        results_super, R_values, thresholds['p0_KS'],
        thresholds['eps_W'], thresholds['eps_TV'], T_max, use_KS_stat
    )
    
    # Compute slopes
    log_R = np.log(mixing_OU['R'])
    slope_theory = 1.0 / mu
    
    fit_KS_OU = np.polyfit(log_R, mixing_OU['KS'], 1)
    fit_W1_OU = np.polyfit(log_R, mixing_OU['W1'], 1)
    fit_TV_OU = np.polyfit(log_R, mixing_OU['TV'], 1)
    
    fit_KS_super = np.polyfit(log_R, mixing_super['KS'], 1)
    fit_W1_super = np.polyfit(log_R, mixing_super['W1'], 1)
    fit_TV_super = np.polyfit(log_R, mixing_super['TV'], 1)
    
    print(f"\nTheoretical OU slope: {slope_theory:.3f}")
    print(f"OU fitted slopes:  KS={fit_KS_OU[0]:.3f}, W₁={fit_W1_OU[0]:.3f}, TV={fit_TV_OU[0]:.3f}")
    print(f"Superlinear fitted slopes:  KS={fit_KS_super[0]:.3f}, W₁={fit_W1_super[0]:.3f}, TV={fit_TV_super[0]:.3f}")
    
    # Log scalar metrics
    wandb.log({
        "slopes/theory": slope_theory,
        "slopes/OU_KS": fit_KS_OU[0],
        "slopes/OU_W1": fit_W1_OU[0],
        "slopes/OU_TV": fit_TV_OU[0],
        "slopes/super_KS": fit_KS_super[0],
        "slopes/super_W1": fit_W1_super[0],
        "slopes/super_TV": fit_TV_super[0],
        "slopes/difference_KS": abs(fit_KS_OU[0] - fit_KS_super[0]),
        "slopes/difference_W1": abs(fit_W1_OU[0] - fit_W1_super[0]),
        "slopes/difference_TV": abs(fit_TV_OU[0] - fit_TV_super[0]),
        "slopes/ratio_KS": fit_KS_super[0] / fit_KS_OU[0] if fit_KS_OU[0] != 0 else 0,
        "slopes/ratio_W1": fit_W1_super[0] / fit_W1_OU[0] if fit_W1_OU[0] != 0 else 0,
        "slopes/ratio_TV": fit_TV_super[0] / fit_TV_OU[0] if fit_TV_OU[0] != 0 else 0
    })
    
    # Log mixing times tables
    table_OU = wandb.Table(columns=["R", "log_R", "KS", "W1", "TV"])
    for i, R in enumerate(R_values):
        table_OU.add_data(R, np.log(R), mixing_OU['KS'][i], mixing_OU['W1'][i], mixing_OU['TV'][i])
    wandb.log({"mixing_times/OU": table_OU})
    
    table_super = wandb.Table(columns=["R", "log_R", "KS", "W1", "TV"])
    for i, R in enumerate(R_values):
        table_super.add_data(R, np.log(R), mixing_super['KS'][i], mixing_super['W1'][i], mixing_super['TV'][i])
    wandb.log({"mixing_times/super": table_super})
    
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    
    # Plot 1: Cutoff shapes for OU
    print("\nCreating cutoff shapes plot for OU...")
    fig = plot_cutoff_shapes(results_OU, R_values, thresholds, "OU Process", use_KS_stat)
    wandb.log({"plots/cutoff_OU": wandb.Image(fig)})
    plt.close(fig)
    
    # Plot 2: Cutoff shapes for Superlinear
    print("Creating cutoff shapes plot for Superlinear...")
    fig = plot_cutoff_shapes(results_super, R_values, thresholds, 
                            f"Superlinear (p={drift_power}, tamed)", use_KS_stat)
    wandb.log({"plots/cutoff_super": wandb.Image(fig)})
    plt.close(fig)
    
    # Plot 3: Combined comparison
    print("Creating combined comparison plot...")
    R_display = R_values[-1]
    fig = plot_combined_comparison(results_OU, results_super, R_display, thresholds, drift_power, use_KS_stat)
    wandb.log({"plots/combined_comparison": wandb.Image(fig)})
    plt.close(fig)
    
    # Plot 4: Asymptotic scaling for OU
    print("Creating asymptotic scaling plot for OU...")
    fig = plot_asymptotic_scaling(mixing_OU, slope_theory, "OU", "blue")
    wandb.log({"plots/asymptotic_OU": wandb.Image(fig)})
    plt.close(fig)
    
    # Plot 5: Asymptotic scaling for Superlinear
    print("Creating asymptotic scaling plot for Superlinear...")
    fig = plot_asymptotic_scaling(mixing_super, None, f"Superlinear (p={drift_power})", "orange")
    wandb.log({"plots/asymptotic_super": wandb.Image(fig)})
    plt.close(fig)
    
    # Plot 6: Combined asymptotic scaling
    print("Creating combined asymptotic scaling plot...")
    fig = plot_combined_asymptotic(mixing_OU, mixing_super, slope_theory, drift_power)
    wandb.log({"plots/asymptotic_combined": wandb.Image(fig)})
    plt.close(fig)
    
    print("\n" + "=" * 80)
    print("DONE - All results logged to WandB")
    print("=" * 80)
    
    # Print WandB run URL
    print(f"\nView results at: {wandb.run.get_url()}")
    
    wandb.finish()


if __name__ == "__main__":
    main()