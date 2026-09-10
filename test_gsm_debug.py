import numpy as np
import trimesh
import manifold3d as m3d
from scipy.interpolate import CubicSpline

stations_x = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000], dtype=float)
sheer_z_pts = np.array([650, 675, 705, 730, 748, 760, 780, 810, 850, 895, 935, 960, 980], dtype=float)
beam_pts = np.array([800, 930, 1040, 1115, 1148, 1155, 1142, 1085, 975, 780, 510, 240, 0], dtype=float)
keel_z_pts = np.array([-50, -120, -195, -255, -295, -320, -310, -270, -200, -110, -15, 350, 980], dtype=float)

cs_sheer = CubicSpline(stations_x, sheer_z_pts)
cs_beam = CubicSpline(stations_x, beam_pts)
cs_keel = CubicSpline(stations_x, keel_z_pts)

nx = 51
x_vals = np.linspace(0, 5990, nx) # end at 5990 so beam > 0, then cap to stem 6000!
n_half = 24

rings = []
for ix, x in enumerate(x_vals):
    b = max(10.0, float(cs_beam(x)))
    kz = float(cs_keel(x))
    sz = float(cs_sheer(x))
    t = x / 6000.0

    if t > 0.5:
        p = 1.25 + 0.55 * (t - 0.5) / 0.5
        q = 1.35 - 0.25 * (t - 0.5) / 0.5
    else:
        p = 1.25 - 0.20 * (0.5 - t) / 0.5
        q = 1.35 + 0.25 * (0.5 - t) / 0.5

    theta_stbd = np.linspace(0, np.pi/2, n_half)
    y_hull_stbd = b * (np.sin(theta_stbd) ** p)
    z_hull_stbd = kz + (sz - kz) * (1.0 - np.cos(theta_stbd) ** q)

    y_deck = np.linspace(b, -b, 15)
    z_deck = []

    if x >= 4000:
        for y in y_deck:
            camber = 40.0 * (1.0 - (y / b)**2)
            z = sz + camber
            if 4350 <= x <= 4850 and abs(y) <= 220:
                z += 30.0
            z_deck.append(z)
    elif x >= 1900:
        cab_w = 0.65 * b
        cab_roof_z = sz + 380.0
        if x >= 3450:
            slope_t = (x - 3450) / 550.0
            cab_roof_z = (1.0 - slope_t) * (sz + 380.0) + slope_t * (sz + 30.0)
            cab_w = (1.0 - slope_t * 0.4) * cab_w
        roof_step_w = 0.50 * cab_w
        pts_y = [b, cab_w + 35, cab_w, cab_w * 0.88, roof_step_w, 0.0, -roof_step_w, -cab_w * 0.88, -cab_w, -cab_w - 35, -b]
        pts_z = [sz, sz + 15, sz + 25, cab_roof_z, cab_roof_z + 18, cab_roof_z + 28, cab_roof_z + 18, cab_roof_z, sz + 25, sz + 15, sz]
        z_interp = np.interp(y_deck, pts_y[::-1], pts_z[::-1])
        z_deck = list(z_interp)
    elif x >= 250:
        coam_top_z = sz + 50.0
        sole_z = 180.0
        bench_z = 450.0
        bench_w = 320.0
        sole_w = 260.0
        coam_w = b - 380.0
        pts_y = [b, coam_w + 50, coam_w, sole_w + bench_w, sole_w, 0.0, -sole_w, -sole_w - bench_w, -coam_w, -coam_w - 50, -b]
        pts_z = [sz, sz + 20, coam_top_z, bench_z, sole_z, sole_z, sole_z, bench_z, coam_top_z, sz + 20, sz]
        z_interp = np.interp(y_deck, pts_y[::-1], pts_z[::-1])
        z_deck = list(z_interp)
    else:
        for y in y_deck:
            camber = 20.0 * (1.0 - (y / b)**2)
            z_deck.append(sz + camber)

    theta_port = np.linspace(np.pi/2, 0, n_half)
    y_hull_port = -b * (np.sin(theta_port) ** p)
    z_hull_port = kz + (sz - kz) * (1.0 - np.cos(theta_port) ** q)

    ring_y = np.concatenate([y_hull_stbd[:-1], y_deck, y_hull_port[1:]])
    ring_z = np.concatenate([z_hull_stbd[:-1], z_deck, z_hull_port[1:]])
    ring_x = np.full_like(ring_y, x)
    rings.append(np.column_stack([ring_x, ring_y, ring_z]))

m_pts = len(rings[0])
all_verts = np.concatenate(rings, axis=0)
all_faces = []

for ix in range(nx - 1):
    base_curr = ix * m_pts
    base_next = (ix + 1) * m_pts
    for im in range(m_pts):
        im_next = (im + 1) % m_pts
        v0 = base_curr + im
        v1 = base_next + im
        v2 = base_next + im_next
        v3 = base_curr + im_next
        all_faces.append([v0, v1, v2])
        all_faces.append([v0, v2, v3])

# Transom cap (x=0)
transom_center = np.array([[0.0, 0.0, 300.0]])
c_idx = len(all_verts)
all_verts = np.vstack([all_verts, transom_center])
for im in range(m_pts):
    im_next = (im + 1) % m_pts
    all_faces.append([c_idx, im_next, im])

# Bow stem cap (x=6000)
bow_center = np.array([[6000.0, 0.0, 980.0]])
b_idx = len(all_verts)
all_verts = np.vstack([all_verts, bow_center])
for im in range(m_pts):
    im_next = (im + 1) % m_pts
    all_faces.append([b_idx, (nx - 1) * m_pts + im, (nx - 1) * m_pts + im_next])

faces_arr = np.array(all_faces)[:, [0, 2, 1]] # flip to outward normals
m = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(all_verts, dtype=np.float32), tri_verts=np.asarray(faces_arr, dtype=np.uint32)))
print("Manifold status:", m.status(), "volume:", m.volume())
tm = trimesh.Trimesh(vertices=m.to_mesh().vert_properties, faces=m.to_mesh().tri_verts)
print("Trimesh watertight:", tm.is_watertight, "volume:", tm.volume)
