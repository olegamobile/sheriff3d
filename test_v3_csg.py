import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import trimesh
import manifold3d as m3d
from scipy.interpolate import CubicSpline

# 1. Hull Splines
stations_x = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000], dtype=float)
sheer_z_pts = np.array([650, 675, 705, 730, 748, 760, 780, 810, 850, 895, 935, 960, 980], dtype=float)
beam_pts = np.array([800, 930, 1040, 1115, 1148, 1155, 1142, 1085, 975, 780, 510, 240, 0], dtype=float)
keel_z_pts = np.array([-50, -120, -195, -255, -295, -320, -310, -270, -200, -110, -15, 350, 980], dtype=float)

cs_sheer = CubicSpline(stations_x, sheer_z_pts)
cs_beam = CubicSpline(stations_x, beam_pts)
cs_keel = CubicSpline(stations_x, keel_z_pts)

# Build Canoe Body (Solid hull capped at sheer)
nx = 81
x_vals = np.linspace(0, 5990, nx)
n_half = 30

rings = []
for ix, x in enumerate(x_vals):
    b = max(4.0, float(cs_beam(x)))
    kz = float(cs_keel(x))
    sz = float(cs_sheer(x))
    t = x / 6000.0
    p = 1.25 + 0.55 * (t - 0.5) / 0.5 if t > 0.5 else 1.25 - 0.20 * (0.5 - t) / 0.5
    q = 1.35 - 0.25 * (t - 0.5) / 0.5 if t > 0.5 else 1.35 + 0.25 * (0.5 - t) / 0.5

    # Starboard hull: from keel (y=0, z=kz) to sheer (y=b, z=sz)
    theta_s = np.linspace(0, np.pi/2, n_half)
    ys = b * (np.sin(theta_s) ** p)
    zs = kz + (sz - kz) * (1.0 - np.cos(theta_s) ** q)

    # Flat sheer cap line from starboard +b to port -b
    y_cap = np.linspace(b, -b, 15)
    z_cap = np.full_like(y_cap, sz)

    # Port hull: from sheer (y=-b, z=sz) to keel (y=0, z=kz)
    theta_p = np.linspace(np.pi/2, 0, n_half)
    yp = -b * (np.sin(theta_p) ** p)
    zp = kz + (sz - kz) * (1.0 - np.cos(theta_p) ** q)

    ring_y = np.concatenate([ys[:-1], y_cap, yp[1:]])
    ring_z = np.concatenate([zs[:-1], z_cap, zp[1:]])
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

tm_hull = trimesh.Trimesh(vertices=all_verts, faces=np.array(all_faces), process=False)
trimesh.repair.fix_winding(tm_hull)
trimesh.repair.fix_normals(tm_hull)
print(f"Hull Canoe Body: Watertight={tm_hull.is_watertight}, Volume={tm_hull.volume:.1f}")

m_hull = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(tm_hull.vertices, dtype=np.float32), tri_verts=np.asarray(tm_hull.faces, dtype=np.uint32)))

# Keel Solid
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
m_keel = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(keel_tm.vertices, dtype=np.float32), tri_verts=np.asarray(keel_tm.faces, dtype=np.uint32)))

# Combine Hull and Keel
m_boat_base = m_hull + m_keel
print("Base Hull + Keel Manifold status:", m_boat_base.status())
