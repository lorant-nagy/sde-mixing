# config.py
"""
Configuration parameters for Sabanis/TUSLA-style taming experiment.
Taming: denominator = 1 + dt^alpha * |X|^(2r)
Regularization: eta * |X|^(2r) * X
"""
import numpy as np

PARAMS = {
    # Basic SDE parameters
    'mu': 0.5,
    'sigma': np.sqrt(1.0),
    
    # Process configurations: list of (alpha, r, p) triplets
    # alpha: power on dt in taming (Sabanis parameter, typically 0.5 or 1)
    # r: state power parameter (taming uses |X|^(2r))
    # p: drift power (drift = -mu * |X|^(p-1) * X)
    #
    # Taming denominator: 1 + dt^alpha * |X|^(2r)
    # Regularization: eta * |X|^(2r) * X
    #
    # Special cases:
    # - (alpha, r=0, p=1): OU-like with constant taming (1 + dt^alpha)
    # - (alpha=0.5, r, p) with 2r≈p: TUSLA-style (matches drift growth)
    # - (alpha=1, r, p): Stronger taming (was original implementation)
    
    'process_configs': [
        (1.0, 0, 1), 
        (1.0, 1, 2),
        (1.0, 2, 3),
    ],
    
    # Simulation parameters
    'T_max': 14.0,
    'N': 50000, # Total number of steps
    'M': 10000,
    'every_k': 10,
    
    # Radii to test
    # 'R_values': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    'R_values': np.array([7.0, 11.0, 15.0, 20.0]),
    
    # Thresholds for mixing time
    'p0_KS': 0.05,
    'eps_W': 0.05,
    'eps_TV': 0.05,
    
    # Stationary reference generation
    'M_stat': 50000,
    
    # WandB settings
    'wandb_project': 'mixing-times_lendulet',
    'wandb_entity': None,
}