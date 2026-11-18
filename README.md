# PI-FM

Code for *Physics-Informed Flow Matching for Distribution System State
Estimation under Low Observability*.

The repository contains the model, the baselines it is compared against, and the
synthetic data path. No datasets, checkpoints or result files are included.

## Install

```
python -m pip install -r requirements.txt
```

Python 3.10 or newer. CUDA is used when available and is not required for the
33-bus system.

## Quick start

```
python -m pifm.pi_fm --system 33 --epochs 200 --meas_rate 0.6 --flow_steps 10
```

Reduced-capacity and physics-loss variants:

```
python -m pifm.pi_fm_lite --system 33
python -m pifm.pi_fm_physloss --system 33
```

Checkpoints and metrics are written to `output/`, created on demand.

## Baselines

```
python -m pifm.baselines.train_single_step --model pigacn --system 33
python -m pifm.baselines.train_single_step --model gcn    --system 33
python -m pifm.baselines.pi_ddpm_res --system 33
python -m pifm.baselines.stdm --system 33
python -m pifm.baselines.wls  --system 33 --method both
python -m pifm.baselines.wlav --system 33
```

`stdm` reads the four-channel tensors from `pifm.data_generator.generate_dataset`
rather than the two-channel state tensor; see its module docstring.

## Layout

| Path | Contents |
|---|---|
| `pifm/pi_fm.py` | PI-FM-Res: flow matching on a residual target, Euler ODE inference |
| `pifm/pi_fm_lite.py`, `pifm/pi_fm_physloss.py` | reduced-capacity and physics-loss variants |
| `pifm/pi_stdm.py` | shared blocks: admittance-modulated attention, denoiser |
| `pifm/distflow_module.py` | DistFlow sweep, producing the coarse estimate |
| `pifm/data_generator.py`, `pifm/dataset_builder.py` | load profiles, power flow, measurement construction |
| `pifm/test_systems.py`, `pifm/feederbw_loader.py` | feeder definitions |
| `pifm/baselines/` | PIGACN, GCN, PI-DDPM-Res, STDM, WLS, WLAV |

## Data

The 33-bus, 118-node and 1746-node systems build their data in memory, so the
commands above run without any input files. The IEEE 123-bus system is the only
one that needs input data:

- `feederbw_loader.build_ieee123_topology(data_dir=...)` reads the three IEEE
  123-bus Excel files (line, switch and spot load data);
- `feederbw_loader.load_feederbw_profiles(data_dir=...)` reads the FeederBW
  measurement parquet files.

Neither is included here, and neither falls back to synthetic loads.

## License

MIT. See `LICENSE`.
