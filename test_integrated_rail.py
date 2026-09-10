import sys
sys.stdout.reconfigure(encoding='utf-8')
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

def get_integrated_ring(x, n_half=24):
    b = max(5.0, float(cs_beam(x)))
    kz = float(cs_keel(x))
    sz = float(cs_sheer(x))
    t = x / 6000.0

    if t > 0.5:
        p = 1.25 + 0.55 * (t - 0.5) / 0.5
        q = 1.35 - 0.25 * (t - 0.5) / 0.5
    else:
        p = 1.25 - 0.20 * (0.5 - t) / 0.5
        q = 1.35 + 0.25 * (0.5 - t) / 0.5

    # 1. Starboard hull (keel to just below sheer)
    theta_stbd = np.linspace(0, np.pi/2 * 0.95, n_half)
    y_hull_stbd = b * (np.sin(theta_stbd) ** p)
    z_hull_stbd = kz + (sz - kz) * (1.0 - np.cos(theta_stbd) ** q)

    # Integrated Rub Rail profile on starboard
    rail_r = 30.0 if b > 50 else b * 0.2
    pts_rail_stbd_y = [b, b + rail_r, b]
    pts_rail_stbd_z = [sz - rail_r*0.8, sz, sz + rail_r*0.6]

    # 2. Deck across from starboard to port
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

    # Integrated Rub Rail on port
    pts_rail_port_y = [-b, -b - rail_r, -b]
    pts_rail_port_z = [sz + rail_r*0.6, sz, sz - rail_r*0.8]

    # 3. Port hull
    theta_port = np.linspace(np.pi/2 * 0.95, 0, n_half)
    y_hull_port = -b * (np.sin(theta_port) ** p)
    z_hull_port = kz + (sz - kz) * (1.0 - np.cos(theta_port) ** q)

    ring_y = np.concatenate([y_hull_stbd, pts_rail_stbd_y, y_deck[1:-1], pts_rail_port_y, y_hull_port[1:]])
    ring_z = np.concatenate([z_hull_stbd, pts_rail_stbd_z, z_deck[1:-1], pts_rail_port_z, z_hull_port[1:]])
    ring_x = np.full_like(ring_y, x)
    return np.column_stack([ring_x, ring_y, ring_z])

nx = 51
x_vals = np.linspace(0, 5990, nx)
rings = [get_integrated_ring(x) for x in x_vals]
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

# Transom cap
c_idx = len(all_verts)
all_verts = np.vstack([all_verts, np.array([[0.0, 0.0, 300.0]])])
for im in range(m_pts):
    im_next = (im + 1) % m_pts
    all_faces.append([c_idx, im_next, im])

# Bow stem cap
b_idx = len(all_verts)
all_verts = np.vstack([all_verts, np.array([[6000.0, 0.0, 980.0]])])
for im in range(m_pts):
    im_next = (im + 1) % m_pts
    all_faces.append([b_idx, (nx - 1) * m_pts + im, (nx - 1) * m_pts + im_next])

tm = trimesh.Trimesh(vertices=all_verts, faces=np.array(all_faces), process=False)
trimesh.repair.fix_winding(tm)
trimesh.repair.fix_normals(tm)
print(f"Integrated Hull+Deck+Rubrail: Watertight={tm.is_watertight}, Volume={tm.volume:.1f} mm³")

# Now Keel solid
keel_stations_z = np.linspace(-260, -850, 16)
keel_verts = []
keel_faces = []
keel_grid = np.zeros((16, 24), dtype=int)
k_count = 0
for iz, z in enumerate(keel_stations_z):
    frac = (z - (-260)) / (-850 - (-260))
    x_le = 3050 - frac * 100
    x_te = 2050 + frac * 200
    chord = x_le - x_te
    max_thick = (95.0 + ((frac - 0.85)/0.15)*90.0) if frac > 0.85 else (95.0 * (1.0 - 0.2*frac))
    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
    for ia, ang in enumerate(angles):
        xc = 0.5 * (1.0 - np.cos(ang))
        yt = 5.0 * (0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*(xc**2) + 0.2843*(xc**3) - 0.1015*(xc**4))
        px = x_te + xc * chord
        py = np.sin(ang) * yt * max_thick
        keel_verts.append([px, py, z])
        keel_grid[iz, ia] = k_count
        k_count += 1

for iz in range(15):
    for ia in range(24):
        v0 = keel_grid[iz, ia]
        v1 = keel_grid[iz + 1, ia]
        v2 = keel_grid[iz + 1, (ia + 1) % 24]
        v3 = keel_grid[iz, (ia + 1) % 24]
        keel_faces.append([v0, v1, v2])
        keel_faces.append([v0, v2, v3])

bulb_bottom = k_count
keel_verts.append([2600.0, 0.0, -855.0])
k_count += 1
for ia in range(24):
    keel_faces.append([bulb_bottom, keel_grid[15, (ia + 1) % 24], keel_grid[15, ia]])

keel_top = k_count
keel_verts.append([2550.0, 0.0, -250.0])
k_count += 1
for ia in range(24):
    keel_faces.append([keel_top, keel_grid[0, ia], keel_grid[0, (ia + 1) % 24]])

keel_tm = trimesh.Trimesh(vertices=np.array(keel_verts), faces=np.array(keel_faces), process=False)
trimesh.repair.fix_winding(keel_tm)
trimesh.repair.fix_normals(keel_tm)
print(f"Keel Watertight={keel_tm.is_watertight}")

# Union with manifold3d
mh = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(tm.vertices, dtype=np.float32), tri_verts=np.asarray(tm.faces, dtype=np.uint32)))
mk = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(keel_tm.vertices, dtype=np.float32), tri_verts=np.asarray(keel_tm.faces, dtype=np.uint32)))
m_full = mh + mk

# Convert back to trimesh
mesh_data = m_full.to_mesh()
full_tm = trimesh.Trimesh(vertices=mesh_data.vert_properties, faces=mesh_data.tri_verts, process=False)
trimesh.repair.fix_winding(full_tm)
trimesh.repair.fix_normals(full_tm)
print(f"FULL SOLID (Hull + Deck + Rubrail + Keel): Watertight={full_tm.is_watertight}, Volume={full_tm.volume:.1f} mm³")
