"""
evaluate_3d.py
--------------
Loads a trained 3D PINN checkpoint and:
  1. Predicts velocity over the near-wall collocation domain and compares
     to the ground-truth CFD field (paper Fig. 6a style).
  2. Computes the 3D wall shear stress VECTOR from the network's velocity
     gradients at the wall and compares its magnitude against COMSOL
     ground truth (paper Fig. 6b/6c) -- never used in training.
  3. Reports relative L2 error, MAE, R^2.

3D WSS (unlike the 2D scalar shortcut used in evaluate.py): with u=0
exactly at a no-slip wall, the viscous traction vector is
    t = (1/Re) * J . n        (J = velocity gradient tensor, n = outward normal)
and the wall shear stress vector is the tangential part of that traction:
    WSS_vec = t - (t . n) n
Its magnitude |WSS_vec| is what's comparable to a scalar CFD WSS export.

REQUIRES a wall WSS ground-truth file (x,y,z,wss) that the paper always
uses for validation but wasn't in your original 4-file list -- export it
from COMSOL the same way you did for the stenosis/2D-aneurysm cases before
running this.

Usage:
    python evaluate_3d.py --data_dir ./normalized --run_dir ./run_3d \\
        --collocation_file whole_cfd_3d_nearwall_norm.csv \\
        --wall_bc_file wall_bc_3d_with_normals_norm.csv \\
        --wss_file wall_wss_3d_norm.csv
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from scipy.spatial import cKDTree

from model_3d import PINN3D
from geometry_utils import crop_to_bbox


def to_tensor(arr, device, requires_grad=False):
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    if requires_grad:
        t.requires_grad_(True)
    return t


def compute_wss_3d(model, x, y, z, nx, ny, nz, device):
    """Returns |WSS| (nondimensional) at each (x,y,z) wall point."""
    xt = to_tensor(x.reshape(-1, 1), device, requires_grad=True)
    yt = to_tensor(y.reshape(-1, 1), device, requires_grad=True)
    zt = to_tensor(z.reshape(-1, 1), device, requires_grad=True)
    _, _, _, _, grads = model.velocity_gradients(xt, yt, zt)
    u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z = [g.detach().cpu().numpy().flatten() for g in grads]

    # traction t = (1/Re) * J . n
    tx = (u_x * nx + u_y * ny + u_z * nz) / model.Re
    ty = (v_x * nx + v_y * ny + v_z * nz) / model.Re
    tz = (w_x * nx + w_y * ny + w_z * nz) / model.Re

    t_dot_n = tx * nx + ty * ny + tz * nz
    wss_x = tx - t_dot_n * nx
    wss_y = ty - t_dot_n * ny
    wss_z = tz - t_dot_n * nz

    return np.sqrt(wss_x ** 2 + wss_y ** 2 + wss_z ** 2)


def relative_l2_error(pred, true):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12)


def subsample(n, max_points, seed=0):
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_points, replace=False)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(os.path.join(args.run_dir, args.checkpoint), map_location=device)
    model = PINN3D(hidden=ckpt["hidden"], n_hidden_layers=ckpt["n_hidden_layers"], Re=ckpt["Re"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint, Re = {ckpt['Re']:.4g}")

    norm_params = ckpt["norm_params"]
    P_c = norm_params["P_c"]
    os.makedirs(args.out_dir, exist_ok=True)

    wss_path = os.path.join(args.data_dir, args.wss_file)
    if not os.path.exists(wss_path):
        raise FileNotFoundError(
            f"\n\n[missing data] Could not find '{args.wss_file}' in {args.data_dir}.\n"
            f"This file (x,y,z,wss ground truth from COMSOL at the wall nodes) wasn't in "
            f"your original 4-file export but is required to validate WSS -- it's the "
            f"whole point of the paper's comparison (Fig. 6b/6c). Export wall-node WSS "
            f"magnitude from COMSOL the same way you did for the stenosis/2D-aneurysm "
            f"cases, normalize it, and point --wss_file at the result.\n"
        )

    # ------------------------------------------------------- 1) velocity field
    cfd = pd.read_csv(os.path.join(args.data_dir, args.collocation_file))
    x = cfd["x"].values.reshape(-1, 1).astype(np.float32)
    y = cfd["y"].values.reshape(-1, 1).astype(np.float32)
    z = cfd["z"].values.reshape(-1, 1).astype(np.float32)
    xmin, xmax = cfd["x"].min(), cfd["x"].max()
    ymin, ymax = cfd["y"].min(), cfd["y"].max()
    zmin, zmax = cfd["z"].min(), cfd["z"].max()

    with torch.no_grad():
        xt, yt, zt = to_tensor(x, device), to_tensor(y, device), to_tensor(z, device)
        u_pred, v_pred, w_pred, _ = model.predict(xt, yt, zt)
        u_pred = u_pred.cpu().numpy().flatten()
        v_pred = v_pred.cpu().numpy().flatten()
        w_pred = w_pred.cpu().numpy().flatten()

    u_true = cfd["u"].values
    v_true = cfd["v"].values
    w_true = cfd["w"].values
    speed_true = np.sqrt(u_true ** 2 + v_true ** 2 + w_true ** 2)
    speed_pred = np.sqrt(u_pred ** 2 + v_pred ** 2 + w_pred ** 2)

    vel_err = relative_l2_error(speed_pred, speed_true)
    print(f"\n[Velocity field] relative L2 error (speed): {vel_err:.4%}")

    plot_idx = subsample(len(x), args.max_plot_points)
    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(121, projection="3d")
    sc0 = ax0.scatter(cfd["x"].values[plot_idx], cfd["y"].values[plot_idx], cfd["z"].values[plot_idx],
                       c=speed_true[plot_idx], cmap="jet", s=2)
    ax0.set_title("Ground truth (CFD) - speed")
    fig.colorbar(sc0, ax=ax0, shrink=0.6)
    ax1 = fig.add_subplot(122, projection="3d")
    sc1 = ax1.scatter(cfd["x"].values[plot_idx], cfd["y"].values[plot_idx], cfd["z"].values[plot_idx],
                       c=speed_pred[plot_idx], cmap="jet", s=2)
    ax1.set_title("PINN prediction - speed")
    fig.colorbar(sc1, ax=ax1, shrink=0.6)
    for a in (ax0, ax1):
        a.set_xlabel("x*"); a.set_ylabel("y*"); a.set_zlabel("z*")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "velocity_comparison_3d.png"), dpi=150)
    print(f"Saved velocity_comparison_3d.png ({len(plot_idx)} of {len(x)} points plotted)")

    # ------------------------------------------------------------- 2) WSS
    wall = pd.read_csv(os.path.join(args.data_dir, args.wall_bc_file))
    wss_gt = pd.read_csv(wss_path)

    for req_col in ("nx", "ny", "nz"):
        if req_col not in wall.columns:
            raise ValueError(f"'{args.wall_bc_file}' has no '{req_col}' column -- 3D WSS "
                              f"needs exact normals, there's no reliable geometry-only "
                              f"fallback for an arbitrary 3D surface like there was for "
                              f"the 2D curved wall.")

    wall, n_drop_wall = crop_to_bbox(wall, xmin, xmax, ymin, ymax, zmin, zmax, margin=args.crop_margin)
    wss_gt, n_drop_wss = crop_to_bbox(wss_gt, xmin, xmax, ymin, ymax, zmin, zmax, margin=args.crop_margin)
    print(f"\nCropped wall_bc to collocation bbox: kept {len(wall)}, dropped {n_drop_wall}")
    print(f"Cropped wss ground truth to collocation bbox: kept {len(wss_gt)}, dropped {n_drop_wss}")

    wall_xyz = wall[["x", "y", "z"]].values
    tree = cKDTree(wall_xyz)
    dist, idx = tree.query(wss_gt[["x", "y", "z"]].values, k=1)
    if dist.max() > 1e-3:
        print(f"[warn] max nearest-neighbor distance between wss ground truth and "
              f"wall_bc points is {dist.max():.3e} (nondim units) -- check the two "
              f"files reference the same wall nodes.")

    nx = wall["nx"].values[idx]
    ny = wall["ny"].values[idx]
    nz = wall["nz"].values[idx]
    xw = wss_gt["x"].values.astype(np.float32)
    yw = wss_gt["y"].values.astype(np.float32)
    zw = wss_gt["z"].values.astype(np.float32)
    wss_true = np.abs(wss_gt["wss"].values)

    wss_pred = compute_wss_3d(model, xw, yw, zw, nx, ny, nz, device)

    wss_err = relative_l2_error(wss_pred, wss_true)
    wss_mae = np.mean(np.abs(wss_pred - wss_true))
    ss_res = np.sum((wss_true - wss_pred) ** 2)
    ss_tot = np.sum((wss_true - np.mean(wss_true)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    wss_mae_dim = wss_mae * P_c

    print(f"\n[WSS] relative L2 error:    {wss_err:.4%}")
    print(f"[WSS] MAE (nondim):          {wss_mae:.4e}")
    print(f"[WSS] MAE (dimensional, Pa): {wss_mae_dim:.4e}")
    print(f"[WSS] R^2:                   {r2:.4f}")

    plot_idx_w = subsample(len(xw), args.max_plot_points)
    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(121, projection="3d")
    sc0 = ax0.scatter(xw[plot_idx_w], yw[plot_idx_w], zw[plot_idx_w],
                       c=wss_true[plot_idx_w], cmap="jet", s=3)
    ax0.set_title("Ground truth (CFD) - |WSS|")
    fig.colorbar(sc0, ax=ax0, shrink=0.6)
    ax1 = fig.add_subplot(122, projection="3d")
    sc1 = ax1.scatter(xw[plot_idx_w], yw[plot_idx_w], zw[plot_idx_w],
                       c=wss_pred[plot_idx_w], cmap="jet", s=3)
    ax1.set_title("PINN prediction - |WSS|")
    fig.colorbar(sc1, ax=ax1, shrink=0.6)
    for a in (ax0, ax1):
        a.set_xlabel("x*"); a.set_ylabel("y*"); a.set_zlabel("z*")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "wss_comparison_3d.png"), dpi=150)
    print(f"Saved wss_comparison_3d.png ({len(plot_idx_w)} of {len(xw)} points plotted)")

    summary = {
        "velocity_relative_L2_error": float(vel_err),
        "wss_relative_L2_error": float(wss_err),
        "wss_MAE_nondimensional": float(wss_mae),
        "wss_MAE_dimensional_Pa": float(wss_mae_dim),
        "wss_R2": float(r2),
        "n_wall_bc_points_after_crop": int(len(wall)),
        "n_wss_points_after_crop": int(len(wss_gt)),
    }
    with open(os.path.join(args.out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved eval_summary.json -> {os.path.join(args.out_dir, 'eval_summary.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./normalized")
    ap.add_argument("--run_dir", type=str, default="./run_3d")
    ap.add_argument("--out_dir", type=str, default="./run_3d/eval")
    ap.add_argument("--checkpoint", type=str, default="pinn_3d_aneurysm.pt")
    ap.add_argument("--collocation_file", type=str, default="whole_cfd_3d_nearwall_norm.csv")
    ap.add_argument("--wall_bc_file", type=str, default="wall_bc_3d_with_normals_norm.csv")
    ap.add_argument("--wss_file", type=str, default="wall_wss_3d_norm.csv",
                     help="You need to export this from COMSOL first -- see docstring.")
    ap.add_argument("--crop_margin", type=float, default=1e-3)
    ap.add_argument("--max_plot_points", type=int, default=20000,
                     help="Subsample scatter plots to this many points (metrics still use all data)")
    args = ap.parse_args()

    main(args)
