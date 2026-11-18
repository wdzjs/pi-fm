"""Baselines compared against in the manuscript.

Each module is self-contained apart from the shared data path and the common
building blocks, which live in the parent package.

    pigacn              PIGACN, admittance-modulated graph attention (Wu et al., 2025)
    train_single_step   training and evaluation for PIGACN and the GCN control
    pi_ddpm_res         PI-DDPM-Res, the same residual formulation under a DDPM objective
    stdm                STDM, spatio-temporal diffusion model (Hua et al., 2026)
    wls                 WLS, Tier A (DistFlow-initialised) and Tier B (flat start)
    wlav                WLAV, IRLS and Huber M-estimator variants
"""
