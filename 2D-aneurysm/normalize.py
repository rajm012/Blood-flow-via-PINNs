"""
normalize.py
------------
Nondimensionalizes every CSV in a folder, auto-detecting each file's
columns rather than assuming fixed filenames. Works unchanged across test
cases (stenosis, aneurysm, ...) as long as columns follow this convention:

    x, y            -- always required, spatial coordinates
    u, v            -- velocity components (if present)
    p               -- pressure (if present)
    wss             -- wall shear stress magnitude (if present)
    nx, ny          -- unit wall-normal components (if present, copied
                       through unchanged since they're already dimensionless)

Characteristic scales (steady, incompressible, Newtonian flow):
    L_c  = channel height H                (length scale)
    U_c  = mean inlet velocity U_avg       (velocity scale)
    P_c  = rho * U_c^2                     (pressure / stress scale)
    Re   = rho * U_c * L_c / mu            (Reynolds number, used inside PINN)

Nondimensional variables:
    x* = x/L_c, y* = y/L_c, u* = u/U_c, v* = v/U_c, p* = p/P_c, wss* = wss/P_c

Every *.csv in --data_dir is processed; each output is written to
--out_dir as <original_basename>_norm.csv, plus a shared norm_params.json.
Files that contain neither u/v/p/wss/nx/ny (i.e. nothing but x,y) are still
normalized (x,y only) so you can keep, e.g., a pure sensor-location file.
"""
import argparse
import glob
import json
import os

import pandas as pd


def normalize_file(in_path, out_dir, L_c, U_c, P_c):
    df = pd.read_csv(in_path)
    if "x" not in df.columns or "y" not in df.columns:
        print(f"  [skip] {in_path}: no x,y columns found")
        return None

    out = pd.DataFrame()
    out["x"] = df["x"] / L_c
    out["y"] = df["y"] / L_c

    detected = ["x", "y"]
    if "u" in df.columns:
        out["u"] = df["u"] / U_c
        detected.append("u")
    if "v" in df.columns:
        out["v"] = df["v"] / U_c
        detected.append("v")
    if "p" in df.columns:
        out["p"] = df["p"] / P_c
        detected.append("p")
    if "wss" in df.columns:
        out["wss"] = df["wss"] / P_c
        detected.append("wss")
    if "nx" in df.columns:
        out["nx"] = df["nx"]  # already unit vector, dimensionless
        detected.append("nx")
    if "ny" in df.columns:
        out["ny"] = df["ny"]
        detected.append("ny")

    base = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(out_dir, f"{base}_norm.csv")
    out.to_csv(out_path, index=False)
    print(f"  [ok] {os.path.basename(in_path)}: {len(df)} rows, columns {detected} -> {out_path}")
    return out_path


def normalize(data_dir, out_dir, rho, mu, H, U_avg):
    os.makedirs(out_dir, exist_ok=True)

    L_c = H
    U_c = U_avg
    P_c = rho * U_c ** 2
    Re = rho * U_c * L_c / mu

    print("Characteristic scales:")
    print(f"  L_c (length)   = {L_c:.6g} m")
    print(f"  U_c (velocity) = {U_c:.6g} m/s")
    print(f"  P_c (pressure) = {P_c:.6g} Pa")
    print(f"  Re             = {Re:.4g}\n")

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in {data_dir}")

    for f in csv_files:
        normalize_file(f, out_dir, L_c, U_c, P_c)

    params = {
        "rho": rho, "mu": mu, "H": H, "U_avg": U_avg,
        "L_c": L_c, "U_c": U_c, "P_c": P_c, "Re": Re,
    }
    with open(os.path.join(out_dir, "norm_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved norm_params.json -> {os.path.join(out_dir, 'norm_params.json')}")
    return params


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".",
                     help="Directory containing the raw COMSOL CSVs")
    ap.add_argument("--out_dir", type=str, default="./normalized",
                     help="Where to write *_norm.csv and norm_params.json")
    ap.add_argument("--rho", type=float, default=1060.0, help="Density [kg/m^3]")
    ap.add_argument("--mu", type=float, default=0.0035, help="Dynamic viscosity [Pa.s]")
    ap.add_argument("--H", type=float, default=2.0, help="Parent-vessel height [m] (length scale)")
    ap.add_argument("--U_avg", type=float, default=0.000124,
                     help="Mean inlet velocity [m/s] (use 0.000264 for the Re~320 aneurysm case)")
    args = ap.parse_args()

    normalize(args.data_dir, args.out_dir, args.rho, args.mu, args.H, args.U_avg)
