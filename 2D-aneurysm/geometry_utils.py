"""
geometry_utils.py
------------------
Small helpers needed once the domain is no longer a flat-walled box (the
aneurysm sac wall is curved, unlike the stenosis case):

  crop_to_bbox          -- restrict points to the region the PINN was
                            actually trained on (paper Sec. III C: the
                            network is only ever asked to represent the
                            cropped domain around the sac, so BC/eval points
                            outside it are meaningless and should be dropped).

  order_by_arclength    -- generic nearest-neighbor path ordering + cumulative
                            arc length "S" along a wall polyline, for plotting
                            WSS vs position the way the paper's Fig. 5(b) does
                            (S along the curved sac wall, not x).

  estimate_wall_normals -- FALLBACK ONLY. Finite-difference tangent from the
                            arc-length ordering, rotated 90 degrees, with sign
                            resolved against a known interior ("fluid side")
                            point. Prefer COMSOL-exported nx,ny when you have
                            them -- this is noisier, especially with uneven
                            point spacing.

Both functions dedupe coincident points internally (common where two COMSOL
boundary segments share a vertex, e.g. the aneurysm sac meeting the parent
vessel wall) -- without this, arc length has a zero-length step at the seam
and the finite-difference tangent divides by zero there.
"""
import numpy as np
from scipy.spatial import cKDTree


def crop_to_bbox(df, xmin, xmax, ymin, ymax, margin=0.0):
    """Returns (df_inside, n_dropped). margin expands the box slightly to
    avoid dropping points that sit exactly on the boundary due to float
    rounding."""
    mask = (
        (df["x"] >= xmin - margin) & (df["x"] <= xmax + margin) &
        (df["y"] >= ymin - margin) & (df["y"] <= ymax + margin)
    )
    n_dropped = int((~mask).sum())
    return df[mask].reset_index(drop=True), n_dropped


def _dedupe_points(x, y, decimals=9):
    """Collapses coincident (x,y) points (within `decimals` precision) into
    unique clusters. Returns:
        unique_idx : original index representing each unique cluster
        inverse    : for each original point, which cluster it belongs to
    so that x[unique_idx][inverse] == x (same for y), i.e. inverse lets you
    broadcast a per-cluster quantity back out to every original point."""
    pts = np.stack([x, y], axis=1)
    coords = np.round(pts, decimals=decimals)
    _, unique_idx, inverse = np.unique(coords, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    return unique_idx, inverse


def _greedy_chain(x, y):
    """Greedy nearest-neighbor path through a set of DISTINCT points
    (no duplicates), starting from the leftmost point. Returns
    (order, s): order is an index permutation, s is cumulative Euclidean
    arc length along that path, aligned with `order`."""
    pts = np.stack([x, y], axis=1)
    n = len(pts)
    if n == 0:
        return np.array([], dtype=int), np.array([])
    if n == 1:
        return np.array([0]), np.array([0.0])

    tree = cKDTree(pts)
    visited = np.zeros(n, dtype=bool)
    start = int(np.argmin(pts[:, 0]))
    order = [start]
    visited[start] = True
    current = start

    for _ in range(n - 1):
        k = 2
        found = None
        while found is None:
            k = min(k, n)
            dist, idx = tree.query(pts[current], k=k)
            idx = np.atleast_1d(idx)
            for j in idx:
                if not visited[j]:
                    found = int(j)
                    break
            if found is None:
                if k >= n:
                    remaining = np.where(~visited)[0]
                    found = int(remaining[0])  # disconnected cluster fallback
                    break
                k *= 2
        order.append(found)
        visited[found] = True
        current = found

    order = np.array(order, dtype=int)
    pts_ordered = pts[order]
    seg = np.sqrt(np.sum(np.diff(pts_ordered, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return order, s


def order_by_arclength(x, y):
    """Orders ALL points (including exact duplicates) by arc length along
    the wall. Duplicates share the same arc-length value and land adjacent
    to each other in the returned order. Returns (order, s) with `order` an
    index permutation into the original x,y arrays and `s[k]` the arc
    length of point `order[k]` -- i.e. use as
        ax.plot(s, values[order])
    exactly as you would with a plain np.argsort."""
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n == 0:
        return np.array([], dtype=int), np.array([])

    unique_idx, inverse = _dedupe_points(x, y)
    ux, uy = x[unique_idx], y[unique_idx]

    order_u, s_u = _greedy_chain(ux, uy)

    s_by_cluster = np.empty(len(ux))
    s_by_cluster[order_u] = s_u

    s_full = s_by_cluster[inverse]
    order_full = np.argsort(s_full, kind="stable")
    s_sorted = s_full[order_full]
    return order_full, s_sorted


def estimate_wall_normals(x, y, fluid_reference_point):
    """FALLBACK when nx,ny were not exported from COMSOL. Estimates unit
    outward normals (pointing away from the fluid, matching the convention
    used in normalize.py / evaluate.py) via finite differences along the
    arc-length-ordered, deduplicated wall polyline, then broadcasts the
    result back out to any duplicate points.

    fluid_reference_point: (x0, y0) any point known to be inside the fluid
    domain (e.g. the centroid of your whole_cfd collocation points) -- used
    only to resolve the sign of the normal (outward vs inward)."""
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 wall points to estimate normals.")

    unique_idx, inverse = _dedupe_points(x, y)
    ux, uy = x[unique_idx], y[unique_idx]
    if len(ux) < 3:
        raise ValueError("Fewer than 3 distinct wall points after deduplication; "
                          "cannot estimate normals.")

    order_u, s_u = _greedy_chain(ux, uy)
    xo, yo = ux[order_u], uy[order_u]

    tx = np.gradient(xo, s_u, edge_order=1)
    ty = np.gradient(yo, s_u, edge_order=1)
    tnorm = np.sqrt(tx ** 2 + ty ** 2)
    tnorm[tnorm == 0] = 1.0
    tx, ty = tx / tnorm, ty / tnorm

    # candidate normal = tangent rotated 90 degrees
    nx_cand, ny_cand = ty, -tx

    fx, fy = fluid_reference_point
    vec_x, vec_y = xo - fx, yo - fy
    dot = nx_cand * vec_x + ny_cand * vec_y
    sign = np.sign(dot)
    sign[sign == 0] = 1.0

    nx_ord, ny_ord = nx_cand * sign, ny_cand * sign

    # map (ordered, deduped) normals back to cluster id, then broadcast to
    # every original point (including duplicates) via `inverse`
    nx_by_cluster = np.empty(len(ux))
    ny_by_cluster = np.empty(len(ux))
    nx_by_cluster[order_u] = nx_ord
    ny_by_cluster[order_u] = ny_ord

    nx_full = nx_by_cluster[inverse]
    ny_full = ny_by_cluster[inverse]
    return nx_full, ny_full
