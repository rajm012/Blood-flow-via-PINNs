# Physics-Informed Neural Networks for Near-Wall Blood Flow

Replication of *"Uncovering near-wall blood flow from sparse data with physics-informed neural networks"*
(Arzani, Wang & D'Souza, Phys. Fluids 33, 071905, 2021).

All CFD data (velocity fields, pressure, wall shear stress, boundary normals) were exported from **COMSOL Multiphysics 6.2**.

---

## Cases

### 2D Stenosis — implemented and run (`2D-stenosis/`)

Replicates paper Sec. III B. Steady 2D incompressible Navier-Stokes in a stenosed channel.

- Re ≈ 75 (mean) / 150 (peak), 5 sparse sensors on the offset surface
- Network: 3 separate MLPs (u, v, p), 9 layers × 128 neurons, Swish
- Training: 5500 epochs, Adam LR 5e-4 → 5e-8, batch 256, λ_d=1, λ_b=20
- Results: velocity field and wall shear stress compared against COMSOL ground truth

```bash
cd 2D-stenosis
python normalize.py --data_dir <raw_csvs> --out_dir normalized --rho 1060 --mu 0.0035 --H 2.0 --U_avg 0.000124
python train_pinn.py --data_dir normalized --out_dir run
python evaluate.py --data_dir normalized --run_dir run --out_dir run/eval
```

---

### 2D Aneurysm — implemented and run (`2D-aneurysm/`)

Replicates paper Sec. III C. Same architecture as stenosis; key differences: Re ≈ 160/320, λ_b=1 (not 20), PINN trained only on a cropped domain around the aneurysm sac (inlet BC unknown).

```bash
cd 2D-aneurysm
python normalize.py --data_dir <raw_csvs> --out_dir normalized --rho 1060 --mu 0.0035 --H 2.0 --U_avg 0.000264
python train_pinn.py --data_dir normalized --out_dir run \
    --collocation_file whole_cfd_aneurysm_norm.csv \
    --wall_bc_file wall_bc_aneurysm_norm.csv \
    --sensor_file five_sensor_aneurysm_norm.csv \
    --lambda_b 1 --epochs 5500
python evaluate.py --data_dir normalized --run_dir run --out_dir run/eval
```

---

### 3D Aneurysm — code and data ready, not run (`3D-aneurysm/`)

Replicates paper Sec. II C 4 / III D. Full 3D near-wall reconstruction from 150 sparse sensors on an offset surface. **Training was not executed due to hardware limitations** (requires a CUDA GPU; ~342k collocation + ~410k wall points with a 9×200 network over 8500 epochs). The code and normalized data are complete and sufficient to replicate the paper's results on appropriate hardware.

- Network: 4 separate MLPs (u, v, w, p), 9 layers × **200 neurons**, Swish
- Training: **8500 epochs**, Adam LR 5e-4 → 5e-7, batch **512**, λ_d=λ_b=**20**
- WSS computed as a proper 3D tangential traction vector (not the 2D scalar shortcut)

**Data files** (in `3D-aneurysm/final_data/`, already normalized):

| File | Points | Contents |
|---|---|---|
| `3d_nearwall_norm.csv` | 342,659 | Collocation points (x,y,z,u,v,w) |
| `3d_150_sensor_data_norm.csv` | 150 | Sparse sensors on offset surface (x,y,z,u,v,w) |
| `3d_wall_bc_normals_norm.csv` | 410,108 | Wall no-slip BC + outward normals (x,y,z,nx,ny,nz) |
| `3d_wall_bc_norm.csv` | 410,108 | Wall WSS ground truth for evaluation (x,y,z,wss) |
| `norm_params.json` | — | Re=159.9, L_c, U_c, P_c |

To replicate:

```bash
cd 3D-aneurysm
python train_pinn_3d.py \
    --data_dir final_data \
    --collocation_file 3d_nearwall_norm.csv \
    --wall_bc_file 3d_wall_bc_normals_norm.csv \
    --sensor_file 3d_150_sensor_data_norm.csv \
    --out_dir run

python evaluate_3d.py \
    --data_dir final_data \
    --collocation_file 3d_nearwall_norm.csv \
    --wall_bc_file 3d_wall_bc_normals_norm.csv \
    --wss_file 3d_wall_bc_norm.csv \
    --run_dir run --out_dir run/eval
```

---

## Reference

A. Arzani, J.-X. Wang, R. M. D'Souza, "Uncovering near-wall blood flow from sparse data with physics-informed neural networks," *Physics of Fluids* **33**, 071905 (2021). https://doi.org/10.1063/5.0055600
