"""
evaluate.py
-----------
Loads a trained PINN checkpoint and:
  1. Predicts velocity over the collocation domain and compares to the
     ground-truth CFD field (paper Fig. 4a / 5a style contour comparison).
  2. Computes wall shear stress (WSS) from the network's velocity gradients
     at the wall and compares against the COMSOL ground truth WSS file
     (paper Fig. 4b / 5b) -- this file is NEVER used in training, only here
     for validation.
  3. Reports quantitative error metrics (relative L2 error, MAE, R^2).

WSS formula (2D, Newtonian, no-slip wall so u=0 exactly on the wall):
    Let J = [[du/dx, du/dy], [dv/dx, dv/dy]]  (velocity gradient tensor)
    directional derivative of u along outward normal n:  d = J . n
    tangential unit vector: t = (-ny, nx)
    nondimensional WSS = (1/Re) * (d . t)
    (consistent with normalize.py's scaling, since P_c = rho*U_c^2 and
     mu*U_c/L_c = P_c/Re)

Curved-wall handling (needed for the aneurysm sac, unlike the flat
stenosis walls):
  - WSS-vs-position is plotted against arc length S along the wall (paper
    Fig. 5b uses "S"), computed generically via geometry_utils.order_by_arclength
    rather than assuming the wall is monotonic in x.
  - If the wall file has no nx,ny columns, normals are estimated from the
    wall geometry itself as a fallback (geometry_utils.estimate_wall_normals)
    -- prefer exporting exact normals from COMSOL when possible; this
    fallback is noisier, especially with uneven point spacing.
  - Both the wall-BC and ground-truth-WSS points are cropped to the
    collocation-point bounding box, since the PINN was only ever trained to
    represent that region (paper Sec. III C, aneurysm case).

Usage (stenosis):
    python evaluate.py --data_dir ./normalized --run_dir ./run \\
        --collocation_file whole_cfd_norm.csv \\
        --wall_bc_file wall_bc_all_norm.csv \\
        --wss_file wall_shear_stress_norm.csv

Usage (aneurysm):
    python evaluate.py --data_dir ./normalized --run_dir ./run \\
        --collocation_file whole_cfd_aneurysm_norm.csv \\
        --wall_bc_file wall_bc_aneurysm_norm.csv \\
        --wss_file wall_wss_aneurysm_norm.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from model import PINN
from geometry_utils import crop_to_bbox, order_by_arclength, estimate_wall_normals


def to_tensor(arr, device, requires_grad=False):
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    if requires_grad:
        t.requires_grad_(True)
    return t


def compute_wss(model, x, y, nx, ny, device):
    """x,y,nx,ny: 1D numpy arrays. Returns nondimensional (signed) WSS."""
    xt = to_tensor(x.reshape(-1, 1), device, requires_grad=True)
    yt = to_tensor(y.reshape(-1, 1), device, requires_grad=True)
    _, _, _, u_x, u_y, v_x, v_y = model.velocity_gradients(xt, yt)

    u_x = u_x.detach().cpu().numpy().flatten()
    u_y = u_y.detach().cpu().numpy().flatten()
    v_x = v_x.detach().cpu().numpy().flatten()
    v_y = v_y.detach().cpu().numpy().flatten()

    d_u = u_x * nx + u_y * ny
    d_v = v_x * nx + v_y * ny
    tx, ty = -ny, nx
    d_dot_t = d_u * tx + d_v * ty
    return (1.0 / model.Re) * d_dot_t


def relative_l2_error(pred, true):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(os.path.join(args.run_dir, args.checkpoint), map_location=device)
    model = PINN(hidden=ckpt["hidden"], n_hidden_layers=ckpt["n_hidden_layers"], Re=ckpt["Re"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint, Re = {ckpt['Re']:.4g}")

    norm_params = ckpt["norm_params"]
    P_c = norm_params["P_c"]

    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------- 1) velocity field
    cfd = pd.read_csv(os.path.join(args.data_dir, args.collocation_file))
    x = cfd["x"].values.reshape(-1, 1).astype(np.float32)
    y = cfd["y"].values.reshape(-1, 1).astype(np.float32)
    xmin, xmax = cfd["x"].min(), cfd["x"].max()
    ymin, ymax = cfd["y"].min(), cfd["y"].max()

    with torch.no_grad():
        xt = to_tensor(x, device)
        yt = to_tensor(y, device)
        u_pred, v_pred, p_pred = model.predict(xt, yt)
        u_pred = u_pred.cpu().numpy().flatten()
        v_pred = v_pred.cpu().numpy().flatten()

    u_true = cfd["u"].values
    v_true = cfd["v"].values
    speed_true = np.sqrt(u_true ** 2 + v_true ** 2)
    speed_pred = np.sqrt(u_pred ** 2 + v_pred ** 2)

    vel_err = relative_l2_error(speed_pred, speed_true)
    print(f"\n[Velocity field] relative L2 error (speed): {vel_err:.4%}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
    sc0 = axes[0].scatter(cfd["x"], cfd["y"], c=speed_true, cmap="jet", s=4)
    axes[0].set_title("Ground truth (CFD) - speed")
    axes[0].set_ylabel("y*")
    fig.colorbar(sc0, ax=axes[0])
    sc1 = axes[1].scatter(cfd["x"], cfd["y"], c=speed_pred, cmap="jet", s=4)
    axes[1].set_title("PINN prediction - speed")
    axes[1].set_xlabel("x*")
    axes[1].set_ylabel("y*")
    fig.colorbar(sc1, ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "velocity_comparison.png"), dpi=150)
    print("Saved velocity_comparison.png")

    # ------------------------------------------------------------- 2) WSS
    wall = pd.read_csv(os.path.join(args.data_dir, args.wall_bc_file))
    wss_gt = pd.read_csv(os.path.join(args.data_dir, args.wss_file))

    wall, n_drop_wall = crop_to_bbox(wall, xmin, xmax, ymin, ymax, margin=args.crop_margin)
    wss_gt, n_drop_wss = crop_to_bbox(wss_gt, xmin, xmax, ymin, ymax, margin=args.crop_margin)
    print(f"\nCropped wall_bc to collocation bbox: kept {len(wall)}, dropped {n_drop_wall}")
    print(f"Cropped wss ground truth to collocation bbox: kept {len(wss_gt)}, dropped {n_drop_wss}")

    have_normals = "nx" in wall.columns and "ny" in wall.columns
    if not have_normals:
        print("\n[warn] wall_bc file has no nx,ny columns. Falling back to "
              "finite-difference normal estimation from wall geometry. "
              "This is noisier than COMSOL-exported normals, especially on "
              "curved walls (e.g. an aneurysm sac) -- re-export nx,ny from "
              "COMSOL for production-quality WSS numbers.")
        fluid_ref = (float(cfd["x"].mean()), float(cfd["y"].mean()))
        nx_all, ny_all = estimate_wall_normals(wall["x"].values, wall["y"].values, fluid_ref)
        wall = wall.copy()
        wall["nx"] = nx_all
        wall["ny"] = ny_all

    # match each ground-truth WSS point to the nearest wall_bc point to
    # obtain its (nx, ny), in case the two files aren't row-aligned
    wall_xy = wall[["x", "y"]].values
    tree = cKDTree(wall_xy)
    dist, idx = tree.query(wss_gt[["x", "y"]].values, k=1)
    if dist.max() > 1e-3:
        print(f"[warn] max nearest-neighbor distance between wss ground truth "
              f"and wall_bc points is {dist.max():.3e} (nondim units). "
              f"Check the two files reference the same wall nodes.")

    nx = wall["nx"].values[idx]
    ny = wall["ny"].values[idx]
    xw = wss_gt["x"].values.astype(np.float32)
    yw = wss_gt["y"].values.astype(np.float32)
    wss_true_mag = np.abs(wss_gt["wss"].values)

    wss_pred = compute_wss(model, xw, yw, nx, ny, device)
    wss_pred_mag = np.abs(wss_pred)

    wss_err = relative_l2_error(wss_pred_mag, wss_true_mag)
    wss_mae = np.mean(np.abs(wss_pred_mag - wss_true_mag))
    ss_res = np.sum((wss_true_mag - wss_pred_mag) ** 2)
    ss_tot = np.sum((wss_true_mag - np.mean(wss_true_mag)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    wss_mae_dim = wss_mae * P_c

    print(f"\n[WSS] relative L2 error:      {wss_err:.4%}")
    print(f"[WSS] MAE (nondim):            {wss_mae:.4e}")
    print(f"[WSS] MAE (dimensional, Pa):   {wss_mae_dim:.4e}")
    print(f"[WSS] R^2:                     {r2:.4f}")

    # arc-length ordering along the wall (works for both flat and curved
    # walls -- generic nearest-neighbor chain, see geometry_utils)
    order, s = order_by_arclength(xw, yw)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(s, wss_true_mag[order], label="CFD WSS", color="tab:blue")
    ax.plot(s, wss_pred_mag[order], "--", label="PINN WSS", color="tab:red")
    ax.set_xlabel("S (arc length along wall, nondimensional)")
    ax.set_ylabel("|WSS| (nondimensional)")
    ax.set_title("Wall shear stress: CFD vs PINN")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "wss_comparison.png"), dpi=150)
    print("Saved wss_comparison.png")

    # ------------------------------------------------------------- summary
    summary = {
        "velocity_relative_L2_error": float(vel_err),
        "wss_relative_L2_error": float(wss_err),
        "wss_MAE_nondimensional": float(wss_mae),
        "wss_MAE_dimensional_Pa": float(wss_mae_dim),
        "wss_R2": float(r2),
        "used_estimated_normals": bool(not have_normals),
        "n_wall_bc_points_after_crop": int(len(wall)),
        "n_wss_points_after_crop": int(len(wss_gt)),
    }
    with open(os.path.join(args.out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved eval_summary.json -> {os.path.join(args.out_dir, 'eval_summary.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./normalized")
    ap.add_argument("--run_dir", type=str, default="./run")
    ap.add_argument("--out_dir", type=str, default="./run/eval")
    ap.add_argument("--checkpoint", type=str, default="pinn_stenosis.pt")
    ap.add_argument("--collocation_file", type=str, default="whole_cfd_norm.csv")
    ap.add_argument("--wall_bc_file", type=str, default="wall_bc_all_norm.csv")
    ap.add_argument("--wss_file", type=str, default="wall_shear_stress_norm.csv")
    ap.add_argument("--crop_margin", type=float, default=1e-3,
                     help="Tolerance (nondim units) added to the crop bbox")
    args = ap.parse_args()

    main(args)
