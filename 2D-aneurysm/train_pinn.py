"""
train_pinn.py
-------------
Trains the 2D stenosis/aneurysm PINN (paper Sec. II C 2/3, III B/C) on
normalized data produced by normalize.py. Filenames are CLI-configurable
since different test cases won't share the same raw filenames.

Loss (paper Eq. 8a):
    L_tot = L_phys + lambda_b * L_BC + lambda_d * L_data

    L_phys : Navier-Stokes residual (momentum x/y + continuity) at the
             collocation-point file (e.g. whole_cfd_norm.csv)
    L_BC   : no-slip (u=v=0) at the wall-BC file (e.g. wall_bc_all_norm.csv)
    L_data : match measured (u, v) at the sensor file (e.g. five_sensor_norm.csv)

Hyperparameters (paper Sec. II C 2/3):
    - 9 hidden layers, 128 neurons/layer, Swish activation, separate nets
      for u, v, p (same architecture for both stenosis and aneurysm cases)
    - Adam optimizer, LR step-decays from 5e-4 to 5e-8 over 5500 epochs
    - batch size 256
    - lambda_d = 1
    - lambda_b = 20 for the stenosis case, lambda_b = 1 for the aneurysm
      case (paper: "All parameters ... selected similar to the previous
      example except lambda_b = 1") -- pass --lambda_b 1 for the aneurysm run

Domain cropping: if the PINN is only defined over a cropped region (paper
Sec. III C, aneurysm case -- inlet BC unknown, so the network is only asked
to represent the sac + immediate surroundings), wall-BC points outside that
region are meaningless to enforce. By default this script crops the wall-BC
file to the bounding box of the collocation points; disable with
--no_crop_wall if you want the raw wall file used as-is (e.g. stenosis case
where collocation already spans the full domain).

Usage (stenosis):
    python train_pinn.py --data_dir ./normalized \\
        --collocation_file whole_cfd_norm.csv \\
        --wall_bc_file wall_bc_all_norm.csv \\
        --sensor_file five_sensor_norm.csv \\
        --lambda_b 20 --epochs 5500

Usage (aneurysm):
    python train_pinn.py --data_dir ./normalized \\
        --collocation_file whole_cfd_aneurysm_norm.csv \\
        --wall_bc_file wall_bc_aneurysm_norm.csv \\
        --sensor_file five_sensor_aneurysm_norm.csv \\
        --lambda_b 1 --epochs 5500
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
from model import PINN
from geometry_utils import crop_to_bbox

def to_tensor(arr, device, requires_grad=False):
    t = torch.tensor(arr, dtype=torch.float32, device=device)
    if requires_grad:
        t.requires_grad_(True)
    return t


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------------- data
    with open(os.path.join(args.data_dir, "norm_params.json")) as f:
        norm_params = json.load(f)
    Re = norm_params["Re"]
    print(f"Loaded norm_params.json -> Re = {Re:.4g}")

    cfd = pd.read_csv(os.path.join(args.data_dir, args.collocation_file))
    wall = pd.read_csv(os.path.join(args.data_dir, args.wall_bc_file))
    sensor = pd.read_csv(os.path.join(args.data_dir, args.sensor_file))

    if not args.no_crop_wall:
        xmin, xmax = cfd["x"].min(), cfd["x"].max()
        ymin, ymax = cfd["y"].min(), cfd["y"].max()
        wall, n_dropped = crop_to_bbox(wall, xmin, xmax, ymin, ymax, margin=args.crop_margin)
        print(f"Cropped wall BC points to collocation bbox "
              f"[{xmin:.4f},{xmax:.4f}] x [{ymin:.4f},{ymax:.4f}]: "
              f"kept {len(wall)}, dropped {n_dropped}")

    x_col = cfd["x"].values.reshape(-1, 1).astype(np.float32)
    y_col = cfd["y"].values.reshape(-1, 1).astype(np.float32)

    x_wall = wall["x"].values.reshape(-1, 1).astype(np.float32)
    y_wall = wall["y"].values.reshape(-1, 1).astype(np.float32)

    x_sens = sensor["x"].values.reshape(-1, 1).astype(np.float32)
    y_sens = sensor["y"].values.reshape(-1, 1).astype(np.float32)
    u_sens = sensor["u"].values.reshape(-1, 1).astype(np.float32)
    v_sens = sensor["v"].values.reshape(-1, 1).astype(np.float32)

    print(f"Collocation points (whole_cfd): {len(x_col)}")
    print(f"Wall BC points (wall_bc_all):   {len(x_wall)}")
    print(f"Sensor points (five_sensor):    {len(x_sens)}  <-- should be 5")

    # sensor set is tiny and fixed -> load once as constant tensors
    x_sens_t = to_tensor(x_sens, device, requires_grad=False)
    y_sens_t = to_tensor(y_sens, device, requires_grad=False)
    u_sens_t = to_tensor(u_sens, device, requires_grad=False)
    v_sens_t = to_tensor(v_sens, device, requires_grad=False)

    # -------------------------------------------------------------- model
    model = PINN(hidden=args.hidden, n_hidden_layers=args.n_hidden_layers, Re=Re).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start)

    # step-decay LR: choose gamma so that after `epochs` epochs, LR ~ lr_end
    n_decay_steps = max(1, args.epochs // args.lr_step_size)
    gamma = (args.lr_end / args.lr_start) ** (1.0 / n_decay_steps)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=gamma)
    print(f"LR schedule: {args.lr_start:.1e} -> ~{args.lr_end:.1e} "
          f"over {args.epochs} epochs (step={args.lr_step_size}, gamma={gamma:.4f})")

    mse = torch.nn.MSELoss()

    n_col = len(x_col)
    n_wall = len(x_wall)
    batch_size = args.batch_size
    steps_per_epoch = max(1, n_col // batch_size)

    history = {"epoch": [], "eqn_loss": [], "bc_loss": [], "data_loss": [], "total_loss": [], "lr": []}

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm_col = np.random.permutation(n_col)
        perm_wall = np.random.permutation(n_wall) if n_wall > 0 else None

        epoch_eqn, epoch_bc, epoch_data = 0.0, 0.0, 0.0

        for step in range(steps_per_epoch):
            # ---- collocation batch (physics loss)
            idx = perm_col[step * batch_size:(step + 1) * batch_size]
            if len(idx) == 0:
                continue
            xb = to_tensor(x_col[idx], device, requires_grad=True)
            yb = to_tensor(y_col[idx], device, requires_grad=True)
            r_mx, r_my, r_c = model.ns_residual(xb, yb)
            L_phys = mse(r_mx, torch.zeros_like(r_mx)) + \
                     mse(r_my, torch.zeros_like(r_my)) + \
                     mse(r_c, torch.zeros_like(r_c))

            # ---- wall BC batch (no-slip)
            if n_wall > 0:
                widx = perm_wall[(step * batch_size) % n_wall: (step * batch_size) % n_wall + batch_size]
                if len(widx) == 0:
                    widx = perm_wall[:min(batch_size, n_wall)]
                xw = to_tensor(x_wall[widx], device, requires_grad=False)
                yw = to_tensor(y_wall[widx], device, requires_grad=False)
                u_w, v_w, _ = model.predict(xw, yw)
                L_bc = mse(u_w, torch.zeros_like(u_w)) + mse(v_w, torch.zeros_like(v_w))
            else:
                L_bc = torch.tensor(0.0, device=device)

            # ---- sparse sensor data (all 5 points every step)
            u_s, v_s, _ = model.predict(x_sens_t, y_sens_t)
            L_data = mse(u_s, u_sens_t) + mse(v_s, v_sens_t)

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
            
            
        # ------------------------------------------------------------- outputs
        os.makedirs(args.out_dir, exist_ok=True)
        checkpoint_dir = os.path.join(args.out_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        if epoch % 500 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"pinn_epoch_{epoch}.pt"
            )

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
            }, checkpoint_path)

            print(f"Checkpoint saved -> {checkpoint_path}")


    ckpt_path = os.path.join(args.out_dir, "pinn_stenosis.pt")
    torch.save({
        "model_state": model.state_dict(),
        "Re": Re,
        "hidden": args.hidden,
        "n_hidden_layers": args.n_hidden_layers,
        "norm_params": norm_params,
    }, ckpt_path)
    print(f"\nSaved checkpoint -> {ckpt_path}")

    hist_df = pd.DataFrame(history)
    hist_path = os.path.join(args.out_dir, "loss_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"Saved loss history -> {hist_path}")

    # convergence plot (paper Fig. 7 style)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.semilogy(hist_df["epoch"], hist_df["eqn_loss"], label="Eqn loss")
    ax.semilogy(hist_df["epoch"], hist_df["bc_loss"], label="BC loss")
    ax.semilogy(hist_df["epoch"], hist_df["data_loss"], label="Data loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("2D stenosis: training convergence")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(args.out_dir, "loss_convergence.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved convergence plot -> {fig_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./normalized")
    ap.add_argument("--out_dir", type=str, default="./run")
    ap.add_argument("--collocation_file", type=str, default="whole_cfd_norm.csv",
                     help="Normalized collocation-point file (physics loss)")
    ap.add_argument("--wall_bc_file", type=str, default="wall_bc_all_norm.csv",
                     help="Normalized wall no-slip file (BC loss)")
    ap.add_argument("--sensor_file", type=str, default="five_sensor_norm.csv",
                     help="Normalized sparse sensor file (data loss)")
    ap.add_argument("--no_crop_wall", action="store_true",
                     help="Disable auto-cropping wall BC points to the collocation bbox")
    ap.add_argument("--crop_margin", type=float, default=1e-3,
                     help="Tolerance (nondim units) added to the crop bbox")
    ap.add_argument("--epochs", type=int, default=5500)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_hidden_layers", type=int, default=9)
    ap.add_argument("--lr_start", type=float, default=5e-4)
    ap.add_argument("--lr_end", type=float, default=5e-8)
    ap.add_argument("--lr_step_size", type=int, default=500,
                     help="epochs between LR step-decays")
    ap.add_argument("--lambda_d", type=float, default=1.0)
    ap.add_argument("--lambda_b", type=float, default=20.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
