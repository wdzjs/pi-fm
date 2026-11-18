"""PI-FM: physics-informed flow matching for distribution system state estimation.

Contains the models, the baselines they are compared against, and the data path.
Datasets, checkpoints and training logs are not part of it.

Modules
    pi_fm            PI-FM-Res, the full model: flow matching on a residual
                     target, conditioned on measurements, with admittance-
                     modulated attention and Euler ODE inference.
    pi_fm_lite       PI-RFM-Lite, the reduced-capacity variant.
    pi_fm_physloss   PI-FM-Res-PL, the variant with an explicit physics loss.
    pi_stdm          shared building blocks: PI attention, denoiser blocks.
    distflow_module  DistFlow backward-forward sweep, used as the physical
                     baseline that the flow model corrects.
    data_generator   synthetic load profiles, DistFlow forward pass, splits.
    dataset_builder  measurement construction: injections, branch flows,
                     coverage mask, noise.
    test_systems     test feeder definitions.
    utils            graph and admittance helpers.
    feederbw_loader  IEEE 123-bus topology with real load profiles.

Subpackages
    baselines        PIGACN, GCN, PI-DDPM-Res, STDM, WLS and WLAV.
"""
