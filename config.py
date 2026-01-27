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
        (0.5, 0, 1),
        (0.5, 1, 2),
        (1.0, 1, 2),
        (1.0, 0.5, 2),
    ],
    
    # Simulation parameters
    'T_max': 7.0,
    'N': 4500, # Total number of steps
    'M': 1000,
    # 'N': 45000, # Total number of steps
    # 'M': 10000,
    'every_k': 10,
    
    # Radii to test
    'R_values': np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
    # 'R_values': np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]),
    
    # Thresholds for mixing time
    'p0_KS': 0.05,
    'eps_W': 0.05,
    'eps_TV': 0.05,
    
    # Stationary reference generation
    'M_stat': 50000,
    
    # WandB settings
    'wandb_project': 'sabanis-tusla-taming',
    'wandb_entity': None,
}