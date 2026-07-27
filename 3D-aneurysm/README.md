# 3D near-wall aneurysm PINN (Arzani, Wang & D'Souza 2021, Sec. II C 4 / III D)

## Files
- `normalize.py` — same generic normalizer as before, now also detects
  `z, w, nz` for 3D data.
- `model_3d.py` — 4 separate MLPs (u, v, w, p), 9×200 architecture (paper's
  3D case uses more neurons/layer than the 128 used for 2D), full 3D NS
  residuals via autograd.
- `geometry_utils.py` — `crop_to_bbox` now also crops on z.
- `train_pinn_3d.py` — paper's 3D hyperparameters: 8500 epochs, LR
  5e-4→5e-7, batch 512, λ_d=λ_b=20. Auto-crops wall points to the
  collocation bounding box (same rationale as 2D case).
- `evaluate_3d.py` — 3D WSS as a proper tangential traction **vector**
  (`WSS = t - (t·n)n`, not the 2D scalar shortcut — that formula doesn't
  generalize to a wall with an arbitrary, continuously-varying 3D normal).
  Requires exact `nx,ny,nz` (no geometry-fallback estimator for 3D — a
  finite-difference tangent-plane estimate on an arbitrary curved surface
  is much less reliable than the 2D single-curve case).

## Run
```bash
python normalize.py --data_dir final_data --out_dir normalized --rho 1060 --mu 0.0035 --H 2.0 --U_avg 0.000264

python train_pinn_3d.py --data_dir final_data --out_dir run --collocation_file 3d_nearwall_norm.csv --wall_bc_file 3d_wall_bc_normals_norm.csv --sensor_file 3d_150_sensor_data_norm.csv

python evaluate_3d.py --data_dir final_data --run_dir run --out_dir run/eval --collocation_file 3d_nearwall_norm.csv --wall_bc_file 3d_wall_bc_normals_norm.csv --wss_file 3d_wall_bc_norm.csv
```
(`--H 2.0` = artery **diameter** = 2×R, matching how L_c was defined for
your earlier 2D cases and giving the same Re≈320 peak-based.)
