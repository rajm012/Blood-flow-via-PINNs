
# # =================================================================

# # import pandas as pd

# # input_file = "3d_outer_surface_bc.csv"
# # output_file = "3d_150_sensor_data.csv"

# # df = pd.read_csv(input_file)
# # sample_df = df.sample(n=150, random_state=42)
# # sample_df.to_csv(output_file, index=False)
# # print(f"Saved {len(sample_df)} rows to '{output_file}'")

# # ===================================================================

# import numpy as np
# import pandas as pd


# CYLINDER_RADIUS = 1.0
# CYLINDER_LENGTH = 10.0
# ANEURYSM_RADIUS = 2.0 
# ANEURYSM_CENTER = np.array([5.0, 1.5, 0.0])
# TOLERANCE = 1e-8

# input_file = '3d_wall_bc.csv'
# output_file = '3d_wall_bc_normals.csv'

# def compute_normal(x, y, z):
#     """
#     Compute the outward unit normal for a point on the wall.
    
#     The geometry consists of:
#     1. Cylinder: y² + z² = R² (outward normal = (0, y/R, z/R))
#     2. Sphere: (x-5)² + (y-1.5)² + z² = R² (outward normal = (x-xc, y-yc, z-zc)/R)
#     """
    
#     dist_from_cylinder_axis = np.sqrt(y**2 + z**2)
#     sphere_vec = np.array([x, y, z]) - ANEURYSM_CENTER
#     dist_from_sphere_center = np.linalg.norm(sphere_vec)
#     on_cylinder = abs(dist_from_cylinder_axis - CYLINDER_RADIUS) < TOLERANCE
#     on_sphere = abs(dist_from_sphere_center - ANEURYSM_RADIUS) < TOLERANCE

#     if on_sphere:
#         normal = sphere_vec / ANEURYSM_RADIUS
#         return normal / np.linalg.norm(normal)
    
#     elif on_cylinder:
#         r = np.sqrt(y**2 + z**2)
#         if r > 0:
#             return np.array([0.0, y/r, z/r])
#         else:
#             return np.array([0.0, 1.0, 0.0])
    
#     else:
#         if dist_from_cylinder_axis < dist_from_sphere_center:
#             r = np.sqrt(y**2 + z**2)
#             if r > 0:
#                 return np.array([0.0, y/r, z/r])
#             else:
#                 return np.array([0.0, 1.0, 0.0])
#         else:
#             normal = sphere_vec / ANEURYSM_RADIUS
#             return normal / np.linalg.norm(normal)


# print(f"Reading data from: {input_file}")
# data = pd.read_csv(input_file)
# points = data[['x', 'y', 'z']].values

# normals = []
# print("Computing normals...")
# for i, (x, y, z) in enumerate(points):
#     n = compute_normal(x, y, z)
#     normals.append(n)

#     if (i + 1) % 100 == 0:
#         print(f"  Processed {i + 1}/{len(points)} points")


# normals = np.array(normals)
# data['nx'] = normals[:, 0]
# data['ny'] = normals[:, 1]
# data['nz'] = normals[:, 2]

# data.to_csv(output_file, index=False)
# print(f"\nSaved with normals to: {output_file}")

# print("\n" + "="*50)
# print("VERIFICATION")
# print("="*50)


# norm_lengths = np.sqrt(data['nx']**2 + data['ny']**2 + data['nz']**2)
# print(f"Normal vector lengths - mean: {norm_lengths.mean():.6f}, std: {norm_lengths.std():.6f}")
# print(f"All normals are unit vectors: {np.allclose(norm_lengths, 1.0, atol=1e-6)}")

# on_cylinder_count = 0
# on_sphere_count = 0
# for x, y, z in points:
#     sphere_vec = np.array([x, y, z]) - ANEURYSM_CENTER
#     dist_from_sphere_center = np.linalg.norm(sphere_vec)
#     dist_from_cylinder_axis = np.sqrt(y**2 + z**2)
#     if abs(dist_from_sphere_center - ANEURYSM_RADIUS) < TOLERANCE:
#         on_sphere_count += 1
#     elif abs(dist_from_cylinder_axis - CYLINDER_RADIUS) < TOLERANCE:
#         on_cylinder_count += 1

# total = len(points)
# print(f"\nSurface classification:")
# print(f"  Cylinder wall:  {on_cylinder_count} points ({on_cylinder_count/total*100:.1f}%)")
# print(f"  Sphere wall:    {on_sphere_count} points ({on_sphere_count/total*100:.1f}%)")
# print(f"  Other:          {total - on_cylinder_count - on_sphere_count} points")


# print("\nFirst 5 rows of the output file:")
# print(data.head())

# try:
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D

#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     step = max(1, len(points) // 500)
#     subset = points[::step]
#     normals_subset = normals[::step]
    
#     ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], s=2, c='blue', alpha=0.5)

#     for i in range(len(subset)):
#         p = subset[i]
#         n = normals_subset[i]
#         ax.quiver(p[0], p[1], p[2], n[0], n[1], n[2], length=0.3, color='red', alpha=0.7)
    
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Wall Points with Outward Normals')
#     plt.savefig('wall_normals_visualization.png', dpi=150)
#     print(f"\nSaved normal visualization to: wall_normals_visualization.png")
#     plt.show()
    
# except ImportError:
#     print("\nMatplotlib not installed. Skipping visualization.")

# print("\nDone!")

# # ===================================================================

import numpy as np

nearwall = np.loadtxt('final_data/3d_wall_bc_normals_norm.csv', delimiter=',')
cropped = nearwall[(nearwall[:, 0] >= 1.5) & (nearwall[:, 0] <= 3.5)]

np.savetxt('3d_wall_bc_normals_norm.csv', cropped, delimiter=',', header='x,y,z,nx,ny,nz', comments='')

print(f"Original: {len(nearwall)} points")
print(f"Cropped:  {len(cropped)} points")


# # ===================================================================
# # ===================================================================
# # ===================================================================


