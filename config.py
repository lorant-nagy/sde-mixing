# config.py
"""
Configuration parameters for OU vs Superlinear Langevin simulation.
"""
import numpy as np

PARAMS = {
    # SDE parameters
    'mu': 1.0,
    'sigma': np.sqrt(2.0),
    'drift_power': 3,  # cubic drift: b(x) = -mu * |x|^(p-1) * x
    
    # Simulation parameters
    'T_max': 15.0,
    'N': 15000,
    'M': 10000,
    'every_k': 10,
    
    # Radii to test
    'R_values': 2.0 ** np.arange(1, 13),  # [2, 4, 8, 16, 32, 64, 128, 256]
    
    # Thresholds for mixing time
    'p0_KS': 0.05,  # Note: p-value threshold (consider using KS statistic instead)
    'eps_W': 0.05,
    'eps_TV': 0.05,
    
    # Stationary reference generation
    'M_stat': 50000,   # Number of stationary samples
    
    # WandB settings
    'wandb_project': 'ou-vs-superlinear',
    'wandb_entity': None,  # Set to your wandb username/team if needed
}