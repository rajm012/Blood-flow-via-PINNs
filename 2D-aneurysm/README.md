# PINN for near-wall blood flow (Arzani, Wang & D'Souza 2021, Phys. Fluids 33, 071905)

Reproduces the paper's PINN methodology for the 2D stenosis and 2D aneurysm
test cases, driven off your COMSOL-exported CSVs.

## Files
- `normalize.py` — nondimensionalizes every CSV in a folder. Auto-detects
  columns (doesn't care about filenames), so it works unchanged across
  test cases.
- `model.py` — 3 separate MLPs (u, v, p), 9 hidden layers × 128 neurons,
  Swish activation; Navier-Stokes residuals via autograd.
- `geometry_utils.py` — bounding-box cropping, arc-length ordering along a
  (possibly curved) wall, and a fallback wall-normal estimator.
- `train_pinn.py` — trains with Adam, LR step-decay 5e-4→5e-8 over 5500
  epochs, batch size 256; λ_d, λ_b, and all filenames are CLI flags.
- `evaluate.py` — computes WSS from the trained network at the wall,
  compares against COMSOL ground truth, reports error metrics, saves plots.

## 1. Normalize
```bash
python normalize.py --data_dir final_data --out_dir  ./normalized --rho 1060 --mu 0.0035 --H 2.0 --U_avg 0.000264
```
Every `*.csv` in `data_dir` gets a `*_norm.csv` twin in `out_dir`, plus a
shared `norm_params.json` (L_c, U_c, P_c, Re) that `train_pinn.py` and
`evaluate.py` read automatically — you never re-enter physical constants
after this step. Check the printed Re against what you expect.

## 2. Aneurysm
Paper differences (Sec. II C 3 / III C): Re≈320, **λ_b = 1** (not 20),
same network architecture, and the PINN is only trained on a **cropped**
domain around the sac (inlet BC unknown → don't ask the network to
reconstruct the entrance region). `--no_crop_wall` is off by default in
`train_pinn.py`, so the wall-BC file is automatically restricted to the
bounding box of your collocation file — this matters more here than for
the stenosis case because your wall/WSS files span the whole domain while
your collocation file (`whole_cfd_aneurysm.csv`) doesn't.

```bash
python train_pinn.py --data_dir normalized --out_dir run --collocation_file whole_cfd_aneurysm_norm.csv --wall_bc_file wall_bc_aneurysm_norm.csv --sensor_file five_sensor_aneurysm_norm.csv --lambda_b 1 --epochs 5500

python evaluate.py --data_dir normalized --run_dir run --out_dir run/eval --collocation_file whole_cfd_norm.csv --wall_bc_file wall_bc_all_norm.csv --wss_file wall_wssnorm.csv --checkpoint pinn_stenosis.pt
```
(The checkpoint filename is always `pinn_stenosis.pt` regardless of which
case you trained — it's just the generic name the training script writes;
harmless, or rename it after training if that bothers you.)

`aneurysm.csv` (the full, uncropped domain) is not fed to either script —
it's only useful for your own reference/visualization of the wider flow
field, since the PINN was never trained to represent the entrance region.

### Wall normals for the curved sac
Your `wall_bc_aneurysm.csv` currently has only `(x, y)` — no `(nx, ny)`.
That's fine for training (no-slip just needs u=v=0), but `evaluate.py`
needs normals to turn velocity gradients into WSS, and a curved wall can't
use a constant `(0,±1)` like the stenosis's flat walls could.

`evaluate.py` will auto-detect the missing columns and fall back to
estimating normals numerically from the wall geometry (finite differences
along the arc-length-ordered wall, sign resolved against the fluid
centroid) — you'll see a `[warn]` printed when this kicks in. It's
serviceable but noisier than an exact normal, especially wherever COMSOL's
wall-node spacing is uneven. **For a final/production run, re-export
`nx, ny` from COMSOL** the same way you did for the stenosis case and pass
that file instead — then `evaluate.py` uses it directly with no fallback.

### Pressure in `five_sensor_aneurysm.csv`
Keep the `p` column for your own comparison, but note the paper's data
loss (Eq. 3, Eq. 7) is velocity-only — `train_pinn.py` only reads `u, v`
from the sensor file regardless of what else is in it, so no change needed
on your end.

## Outputs
**Training** (`<out_dir>/`): `pinn_stenosis.pt` (checkpoint), `loss_history.csv`,
`loss_convergence.png` (paper Fig. 7 style).

**Evaluation** (`<out_dir>/eval/`): `velocity_comparison.png` (paper Fig.
4a/5a style), `wss_comparison.png` — plotted against arc length `S` along
the wall rather than raw `x`, so it works whether the wall is flat
(stenosis) or curved (aneurysm) — and `eval_summary.json` with relative L2
error, MAE (nondimensional + Pa), and R² for WSS, plus relative L2 error
for the velocity field.
