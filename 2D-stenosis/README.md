# 2D Stenosis PINN (reproduces Arzani, Wang & D'Souza 2021, Sec. III B)

Implements the PINN described in *"Uncovering near-wall blood flow from
sparse data with physics-informed neural networks"* (Phys. Fluids 33,
071905, 2021) for the 2D stenosis test case, using your COMSOL-exported
CSVs.

## 1. Put your raw CSVs in one folder
```
whole_cfd.csv          x, y, u, v, p
wall_bc_all.csv        x, y, nx, ny
five_sensor.csv        x, y, u, v
wall_shear_stress.csv  x, y, wss
```

## 2. Normalize (nondimensionalize)
```bash
pip install -r requirements.txt

python normalize.py \
    --data_dir /path/to/raw_csvs \
    --out_dir  ./normalized \
    --rho 1060 --mu 0.0035 --H 2.0 --U_avg 0.000124
```
This writes `*_norm.csv` for each file plus `norm_params.json`
(scales L_c, U_c, P_c, and Re — reused automatically by the next two
scripts, so you never have to re-enter them).

Sanity check the printed Re: with your numbers it should come out to
Re ≈ 75 (mean-velocity based) — consistent with Re ≈ 150 peak-based,
matching the paper.

## 3. Train
```bash
python train_pinn.py --data_dir ./normalized --out_dir ./run
```
Defaults match the paper's 2D-stenosis hyperparameters (Sec. II C 2):
9 hidden layers × 128 neurons, Swish activation, 3 separate networks
(u, v, p), Adam with LR step-decaying 5e-4 → 5e-8 over 5500 epochs,
batch size 256, λ_d=1, λ_b=20. All overridable via CLI flags — run
`python train_pinn.py -h` to see them.

Outputs in `./run/`:
- `pinn_stenosis.pt` — checkpoint (weights + Re + norm_params, so eval
  never needs you to re-supply physical constants)
- `loss_history.csv`, `loss_convergence.png` — Eqn/BC/Data loss vs
  epoch (paper Fig. 7 style)

## 4. Evaluate against COMSOL ground truth
```bash
python evaluate.py --data_dir ./normalized --run_dir ./run --out_dir ./run/eval
```
`wall_shear_stress.csv` is **only** read here, never during training —
matching the paper's use of it purely for validation.

Outputs in `./run/eval/`:
- `velocity_comparison.png` — CFD vs PINN speed contours (paper Fig. 4a)
- `wss_comparison.png` — CFD vs PINN |WSS| along the wall (paper Fig. 4b)
- `eval_summary.json` — relative L2 error, MAE (nondim + Pa), R² for WSS,
  and relative L2 error for the velocity field

## Notes / things worth checking on your actual data
- **WSS sign convention**: the script compares `|WSS|` (magnitude) on both
  sides, since sign convention depends on wall orientation and how COMSOL
  exported it. If you need signed WSS, check `wall_shear_stress.csv`'s
  sign convention and adjust `compute_wss()` in `evaluate.py` (tangent
  direction `t = (-ny, nx)` vs `(ny, -nx)`).
- **Arc-length axis**: `evaluate.py` currently sorts wall points by `x*`
  for the WSS-vs-position plot. That's fine if your wall is a single
  roughly-monotonic-in-x boundary (e.g. bottom wall only). If
  `wall_shear_stress.csv` contains both top and bottom walls together,
  either filter to one wall first or replace the sort key with true
  cumulative arc-length.
- **Nearest-neighbor matching**: `evaluate.py` matches each
  `wall_shear_stress` point to its nearest `wall_bc_all` point to borrow
  its normal vector. It prints a warning if that match isn't tight —
  worth checking that warning doesn't fire on your real data (it would
  mean the two exports aren't on the same wall nodes).
- **GPU**: both `train_pinn.py` and `evaluate.py` auto-detect CUDA; runs
  on CPU otherwise (just slower).
