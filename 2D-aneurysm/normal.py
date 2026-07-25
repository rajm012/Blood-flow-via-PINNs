
import numpy as np

def calculate_normals(wall_points):
    circles = [
        {'xc': 5.0, 'yc': 3.0, 'R': 1.5},
        {'xc': 5.0, 'yc': -1.0, 'R': 1.5}
    ]

    top_y = 2.0
    bottom_y = 0.0
    tol = 1e-5

    N = wall_points.shape[0]
    normals = np.zeros((N, 2))

    for i, (x, y) in enumerate(wall_points):
        matched = False
        for circ in circles:
            xc, yc, R = circ['xc'], circ['yc'], circ['R']
            dist = np.sqrt((x - xc)**2 + (y - yc)**2)
            if abs(dist - R) < tol:
                nx = -(x - xc) / R
                ny = -(y - yc) / R
                normals[i, :] = [nx, ny]
                matched = True
                break

        if not matched:
            if abs(y - top_y) < tol:
                normals[i, :] = [0.0, 1.0]
            elif abs(y - bottom_y) < tol:
                normals[i, :] = [0.0, -1.0]
            else:
                print(f"Warning: Point ({x:.4f}, {y:.4f}) not matched. Guessing based on y.")
                normals[i, :] = [0.0, 1.0] if y > 0 else [0.0, -1.0]

    return normals


wall_data = np.loadtxt('final_data/wall_bc_aneurysm.csv', delimiter=',')
points = wall_data[:, 0:2]

normals = calculate_normals(points)
new_wall_data = np.hstack((points, normals))

np.savetxt('final_data/wall_bc_with_normals.csv', new_wall_data, header='x,y,nx,ny', delimiter=',', comments='')
print(f"   Processed {len(points)} wall points.")
