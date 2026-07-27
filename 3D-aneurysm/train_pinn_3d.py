"""
train_pinn_3d.py
-----------------
Trains the 3D near-wall aneurysm PINN (paper Sec. II C 4 / III D) on
normalized data.

Loss (paper Eq. 8a):
    L_tot = L_phys + lambda_b * L_BC + lambda_d * L_data

    L_phys : 3D Navier-Stokes residual (momentum x/y/z + continuity) at the
             near-wall collocation-point file
    L_BC   : no-slip (u=v=w=0) at the wall file (uses only x,y,z -- normals
             aren't needed for training, only for WSS in evaluate_3d.py)
    L_data : match measured (u, v, w) at the SPARSE sensor file

IMPORTANT: the paper's 3D case supplies data only through a small number
of sparse sensors on the offset ("outer") surface (125 in the paper; your
`sensors_3d_150.csv` is the equivalent) -- NOT the full, densely-sampled
offset surface. This script intentionally does not read a dense
"outer_surface_bc" file. If you feed the dense file in as an additional
BC/data source, you are no longer replicating the paper's sparse-data
result -- you've turned it into a much better-posed (and much easier)
problem. Do that deliberately if at all, not by accident.

Hyperparameters (paper Sec. II C 4, 3D aneurysm case -- LARGER network and
LONGER training than the 2D cases):
    - 9 hidden layers, 200 neurons/layer (vs 128 for 2D), Swish activation,
      4 separate nets (u, v, w, p)
    - Adam optimizer, LR step-decays from 5e-4 to 5e-7 over 8500 epochs
      (2D cases decay to 5e-8; the 3D case only goes to 5e-7)
    - batch size 512 (vs 256 for 2D)
    - lambda_d = 20, lambda_b = 20 (both larger than either 2D case)

Domain cropping: if your near-wall extraction was NOT already axially
cropped around the aneurysm (paper Fig. 2c shows a local segment, not the
full 10 m artery), the wall-BC file may span more of the domain than your
collocation points do. This script crops wall points to the collocation
bounding box by default, same rationale as the 2D case -- disable with
--no_crop_wall.

Usage:
    python train_pinn_3d.py --data_dir ./normalized \\
        --collocation_file whole_cfd_3d_nearwall_norm.csv \\
        --wall_bc_file wall_bc_3d_with_normals_norm.csv \\
        --sensor_file sensors_3d_150_norm.csv \\
        --lambda_b 20 --lambda_d 20 --epochs 8500 --batch_size 512
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_3d import PINN3D
from geometry_utils import crop_to_bbox


def to_tensor(arr, device, requires_grad=False):
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    if requires_grad:
        t.requires_grad_(True)
    return t


def save_checkpoint(path, model, optimizer, scheduler, Re, args, norm_params, history, epoch):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "Re": Re,
        "hidden": args.hidden,
        "n_hidden_layers": args.n_hidden_layers,
        "norm_params": norm_params,
        "history": history,
    }, path)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("[warn] No GPU detected. With ~500k-1M points and a 9x200 "
              "network this will be very slow on CPU -- strongly consider "
              "running this on a machine with a CUDA GPU.")

    with open(os.path.join(args.data_dir, "norm_params.json")) as f:
        norm_params = json.load(f)
    Re = norm_params["Re"]
    print(f"Loaded norm_params.json -> Re = {Re:.4g}")

    cfd = pd.read_csv(os.path.join(args.data_dir, args.collocation_file))
    wall = pd.read_csv(os.path.join(args.data_dir, args.wall_bc_file))
    sensor = pd.read_csv(os.path.join(args.data_dir, args.sensor_file))

    for name, df in [("collocation", cfd), ("wall", wall), ("sensor", sensor)]:
        if "z" not in df.columns:
            raise ValueError(f"{name} file has no 'z' column -- is this "
                              f"actually 3D data? Re-run normalize.py if "
                              f"you just added z,w,nz support.")

    if not args.no_crop_wall:
        xmin, xmax = cfd["x"].min(), cfd["x"].max()
        ymin, ymax = cfd["y"].min(), cfd["y"].max()
        zmin, zmax = cfd["z"].min(), cfd["z"].max()
        wall, n_dropped = crop_to_bbox(wall, xmin, xmax, ymin, ymax, zmin, zmax, margin=args.crop_margin)
        print(f"Cropped wall BC points to collocation bbox: kept {len(wall)}, dropped {n_dropped}")

    x_col = cfd["x"].values.reshape(-1, 1).astype(np.float32)
    y_col = cfd["y"].values.reshape(-1, 1).astype(np.float32)
    z_col = cfd["z"].values.reshape(-1, 1).astype(np.float32)

    x_wall = wall["x"].values.reshape(-1, 1).astype(np.float32)
    y_wall = wall["y"].values.reshape(-1, 1).astype(np.float32)
    z_wall = wall["z"].values.reshape(-1, 1).astype(np.float32)

    x_sens = sensor["x"].values.reshape(-1, 1).astype(np.float32)
    y_sens = sensor["y"].values.reshape(-1, 1).astype(np.float32)
    z_sens = sensor["z"].values.reshape(-1, 1).astype(np.float32)
    u_sens = sensor["u"].values.reshape(-1, 1).astype(np.float32)
    v_sens = sensor["v"].values.reshape(-1, 1).astype(np.float32)
    w_sens = sensor["w"].values.reshape(-1, 1).astype(np.float32)

    print(f"Collocation points: {len(x_col)}")
    print(f"Wall BC points:     {len(x_wall)}")
    print(f"Sensor points:      {len(x_sens)}  <-- should be ~125-150")

    x_sens_t = to_tensor(x_sens, device)
    y_sens_t = to_tensor(y_sens, device)
    z_sens_t = to_tensor(z_sens, device)
    u_sens_t = to_tensor(u_sens, device)
    v_sens_t = to_tensor(v_sens, device)
    w_sens_t = to_tensor(w_sens, device)

    model = PINN3D(hidden=args.hidden, n_hidden_layers=args.n_hidden_layers, Re=Re).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start)

    n_decay_steps = max(1, args.epochs // args.lr_step_size)
    gamma = (args.lr_end / args.lr_start) ** (1.0 / n_decay_steps)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=gamma)
    print(f"LR schedule: {args.lr_start:.1e} -> ~{args.lr_end:.1e} over {args.epochs} epochs")

    mse = torch.nn.MSELoss()

    n_col = len(x_col)
    n_wall = len(x_wall)
    batch_size = args.batch_size
    steps_per_epoch = max(1, n_col // batch_size)
    print(f"Steps per epoch: {steps_per_epoch} (total epochs: {args.epochs} "
          f"-> ~{steps_per_epoch * args.epochs} optimizer steps)")

    history = {"epoch": [], "eqn_loss": [], "bc_loss": [], "data_loss": [], "total_loss": [], "lr": []}

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm_col = np.random.permutation(n_col)
        perm_wall = np.random.permutation(n_wall) if n_wall > 0 else None

        epoch_eqn, epoch_bc, epoch_data = 0.0, 0.0, 0.0

        for step in range(steps_per_epoch):
            idx = perm_col[step * batch_size:(step + 1) * batch_size]
            if len(idx) == 0:
                continue
            xb = to_tensor(x_col[idx], device, requires_grad=True)
            yb = to_tensor(y_col[idx], device, requires_grad=True)
            zb = to_tensor(z_col[idx], device, requires_grad=True)
            r_mx, r_my, r_mz, r_c = model.ns_residual(xb, yb, zb)
            L_phys = mse(r_mx, torch.zeros_like(r_mx)) + \
                     mse(r_my, torch.zeros_like(r_my)) + \
                     mse(r_mz, torch.zeros_like(r_mz)) + \
                     mse(r_c, torch.zeros_like(r_c))

            if n_wall > 0:
                widx = perm_wall[(step * batch_size) % n_wall: (step * batch_size) % n_wall + batch_size]
                if len(widx) == 0:
                    widx = perm_wall[:min(batch_size, n_wall)]
                xw = to_tensor(x_wall[widx], device)
                yw = to_tensor(y_wall[widx], device)
                zw = to_tensor(z_wall[widx], device)
                u_w, v_w, w_w, _ = model.predict(xw, yw, zw)
                L_bc = mse(u_w, torch.zeros_like(u_w)) + \
                       mse(v_w, torch.zeros_like(v_w)) + \
                       mse(w_w, torch.zeros_like(w_w))
            else:
                L_bc = torch.tensor(0.0, device=device)

            u_s, v_s, w_s, _ = model.predict(x_sens_t, y_sens_t, z_sens_t)
            L_data = mse(u_s, u_sens_t) + mse(v_s, v_sens_t) + mse(w_s, w_sens_t)

            L_tot = L_phys + args.lambda_b * L_bc + args.lambda_d * L_data

            optimizer.zero_grad()
            L_tot.backward()
            optimizer.step()

            epoch_eqn += L_phys.item()
            epoch_bc += L_bc.item()
            epoch_data += L_data.item()

        scheduler.step()

        epoch_eqn /= steps_per_epoch
        epoch_bc /= steps_per_epoch
        epoch_data /= steps_per_epoch
        epoch_tot = epoch_eqn + args.lambda_b * epoch_bc + args.lambda_d * epoch_data

        history["epoch"].append(epoch)
        history["eqn_loss"].append(epoch_eqn)
        history["bc_loss"].append(epoch_bc)
        history["data_loss"].append(epoch_data)
        history["total_loss"].append(epoch_tot)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if epoch % args.log_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"epoch {epoch:5d}/{args.epochs} | eqn {epoch_eqn:.3e} | "
                  f"bc {epoch_bc:.3e} | data {epoch_data:.3e} | tot {epoch_tot:.3e} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        if epoch % 500 == 0:
            checkpoint_dir = os.path.join(args.out_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"pinn_epoch_{epoch}.pt")
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, Re, args, norm_params, history, epoch)
            print(f"Checkpoint saved -> {checkpoint_path}")

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_path = os.path.join(args.out_dir, "pinn_3d_aneurysm.pt")
    save_checkpoint(ckpt_path, model, optimizer, scheduler, Re, args, norm_params, history, args.epochs)
    print(f"\nSaved checkpoint -> {ckpt_path}")

    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(args.out_dir, "loss_history.csv")
    hist_df.to_csv(hist_path, index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.semilogy(hist_df["epoch"], hist_df["eqn_loss"], label="Eqn loss")
    ax.semilogy(hist_df["epoch"], hist_df["bc_loss"], label="BC loss")
    ax.semilogy(hist_df["epoch"], hist_df["data_loss"], label="Data loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("3D near-wall aneurysm: training convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "loss_convergence.png"), dpi=150)
    print(f"Saved loss_history.csv and loss_convergence.png -> {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./normalized")
    ap.add_argument("--out_dir", type=str, default="./run_3d")
    ap.add_argument("--collocation_file", type=str, default="whole_cfd_3d_nearwall_norm.csv")
    ap.add_argument("--wall_bc_file", type=str, default="wall_bc_3d_with_normals_norm.csv")
    ap.add_argument("--sensor_file", type=str, default="sensors_3d_150_norm.csv")
    ap.add_argument("--no_crop_wall", action="store_true")
    ap.add_argument("--crop_margin", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=8500)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--n_hidden_layers", type=int, default=9)
    ap.add_argument("--lr_start", type=float, default=5e-4)
    ap.add_argument("--lr_end", type=float, default=5e-7)
    ap.add_argument("--lr_step_size", type=int, default=500)
    ap.add_argument("--lambda_d", type=float, default=20.0)
    ap.add_argument("--lambda_b", type=float, default=20.0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
