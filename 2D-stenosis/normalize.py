
"""
normalize.py
------------
Nondimensionalizes the COMSOL-exported CSVs so they are on O(1) scales
before going into the PINN (paper: "All geometric dimensions and variables
were normalized, which facilitates convergence in NNs.").

Characteristic scales used (steady, incompressible, Newtonian flow):
    L_c  = channel height H                (length scale)
    U_c  = mean inlet velocity U_avg       (velocity scale)
    P_c  = rho * U_c^2                     (pressure / stress scale)
    Re   = rho * U_c * L_c / mu            (Reynolds number, used inside PINN)

Nondimensional variables:
    x* = x / L_c ,  y* = y / L_c
    u* = u / U_c ,  v* = v / U_c
    p* = p / P_c
    wss* = wss / P_c        (same scaling as pressure, since stress ~ Pa)

Inputs (raw, dimensional, as produced by COMSOL):
    whole_cfd.csv          columns: x, y, u, v, p
    wall_bc_all.csv         columns: x, y, nx, ny        (nx, ny already unit normals -> untouched)
    five_sensor.csv        columns: x, y, u, v
    wall_shear_stress.csv  columns: x, y, wss

Outputs (written next to the inputs, same directory):
    *_norm.csv for each of the four files above
    norm_params.json  -> L_c, U_c, P_c, rho, mu, Re, plus paths, for reuse
                          in train_pinn.py and evaluate.py.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd


def normalize(data_dir, out_dir, rho, mu, H, U_avg):
    os.makedirs(out_dir, exist_ok=True)
    L_c = H
    U_c = U_avg
    P_c = rho * U_c ** 2
    Re = rho * U_c * L_c / mu

    print(f"Characteristic scales:")
    print(f"  L_c (length)   = {L_c:.6g} m")
    print(f"  U_c (velocity) = {U_c:.6g} m/s")
    print(f"  P_c (pressure) = {P_c:.6g} Pa")
    print(f"  Re             = {Re:.4g}")

    files = {
        "whole_cfd": ("x", "y", "u", "v", "p"),
        "wall_bc_all": ("x", "y", "nx", "ny"),
        "five_sensor": ("x", "y", "u", "v"),
        "wall_shear_stress": ("x", "y", "wss"),
    }

    for name, cols in files.items():
        in_path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(in_path):
            print(f"  [skip] {in_path} not found")
            continue

        df = pd.read_csv(in_path)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name}.csv is missing expected columns {missing}. "
                              f"Found columns: {list(df.columns)}")

        out = pd.DataFrame()
        out["x"] = df["x"] / L_c
        out["y"] = df["y"] / L_c

        if "u" in df.columns:
            out["u"] = df["u"] / U_c
        if "v" in df.columns:
            out["v"] = df["v"] / U_c
        if "p" in df.columns:
            out["p"] = df["p"] / P_c
        if "wss" in df.columns:
            out["wss"] = df["wss"] / P_c
        if "nx" in df.columns:
            out["nx"] = df["nx"]  # already unit vector, dimensionless
        if "ny" in df.columns:
            out["ny"] = df["ny"]

        out_path = os.path.join(out_dir, f"{name}_norm.csv")
        out.to_csv(out_path, index=False)
        print(f"  [ok] {name}: {len(df)} rows -> {out_path}")

    params = {
        "rho": rho,
        "mu": mu,
        "H": H,
        "U_avg": U_avg,
        "L_c": L_c,
        "U_c": U_c,
        "P_c": P_c,
        "Re": Re
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
    ap.add_argument("--H", type=float, default=2.0, help="Channel height [m] (length scale)")
    ap.add_argument("--U_avg", type=float, default=0.000124, help="Mean inlet velocity [m/s]")
    args = ap.parse_args()
    normalize(args.data_dir, args.out_dir, args.rho, args.mu, args.H, args.U_avg)


