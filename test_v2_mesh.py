import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import trimesh
import manifold3d as m3d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Calibration splines
stations_x = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000], dtype=float)
sheer_z_pts = np.array([650, 675, 705, 730, 748, 760, 780, 810, 850, 895, 935, 960, 980], dtype=float)
beam_pts = np.array([800, 930, 1040, 1115, 1148, 1155, 1142, 1085, 975, 780, 510, 240, 0], dtype=float)
keel_z_pts = np.array([-50, -120, -195, -255, -295, -320, -310, -270, -200, -110, -15, 350, 980], dtype=float)

cs_sheer = CubicSpline(stations_x, sheer_z_pts)
cs_beam = CubicSpline(stations_x, beam_pts)
cs_keel = CubicSpline(stations_x, keel_z_pts)

def get_station_ring_v2(x, n_hull_half=35, n_deck_half=35):
    b = max(4.0, float(cs_beam(x)))
    kz = float(cs_keel(x))
    sz = float(cs_sheer(x))
    t = x / 6000.0

    # Hull flare and deadrise parameters
    if t > 0.5:
        p = 1.25 + 0.55 * (t - 0.5) / 0.5
        q = 1.35 - 0.25 * (t - 0.5) / 0.5
    else:
        p = 1.25 - 0.20 * (0.5 - t) / 0.5
        q = 1.35 + 0.25 * (0.5 - t) / 0.5

    # 1. Starboard underwater / topside hull
    theta_stbd = np.linspace(0, np.pi/2 * 0.96, n_hull_half)
    y_hull_stbd = b * (np.sin(theta_stbd) ** p)
    z_hull_stbd = kz + (sz - kz) * (1.0 - np.cos(theta_stbd) ** q)

    # Integrated smooth Rub rail bead at sheer
    rail_r = 25.0 if b > 50 else b * 0.15
    pts_rail_stbd_y = [b, b + rail_r, b]
    pts_rail_stbd_z = [sz - rail_r*0.7, sz, sz + rail_r*0.5]

    # 2. Deck across: we define starboard half from 0 (centerline) to b (sheer)
    y_deck_eval = np.linspace(0.0, b, n_deck_half)
    
    if x >= 4150:
        # Foredeck: clean smooth cambered surface with hatch
        camber = 35.0 * (1.0 - (y_deck_eval / max(1.0, b))**2)
        z_deck_half = sz + camber
        # Forward Hatch (Panneau ouvrant) on foredeck: x in 4250..4750
        if 4250 <= x <= 4750:
            hatch_w = 230.0 - 50.0 * (x - 4250) / 500.0
            for i, y in enumerate(y_deck_eval):
                if y <= hatch_w:
                    z_deck_half[i] += 18.0 # clean raised hatch frame

    elif x >= 2050:
        # Coachroof (Рубка):
        # Forward slope: from x=3350 to x=4150, a perfectly flat/smooth slope down to foredeck
        if x >= 3350:
            slope_t = (x - 3350) / 800.0 # 0 at 3350, 1 at 4150
            roof_h = (1.0 - slope_t) * 360.0 + slope_t * 25.0
            cab_w = (1.0 - slope_t * 0.20) * (0.70 * b)
        else:
            roof_h = 360.0
            cab_w = 0.70 * b

        cab_roof_z = sz + roof_h
        side_deck_w = b - cab_w

        # Symmetrical smooth coachroof profile (0 = centerline, cab_w = cabin side, b = sheer)
        key_y = np.array([0.0, cab_w * 0.85, cab_w, cab_w + 30.0, b])
        key_z = np.array([cab_roof_z + 20.0, cab_roof_z, sz + 20.0, sz + 12.0, sz])
        z_deck_half = np.interp(y_deck_eval, key_y, key_z)

        # Forward hatch window if x is on the forward slope (x in 3650..4100)
        if 3650 <= x <= 4100:
            hatch_w = 230.0
            for i, y in enumerate(y_deck_eval):
                if y <= hatch_w:
                    z_deck_half[i] += 16.0

    elif x >= 350:
        # Cockpit with real benches and sunken footwell!
        sole_w = 220.0 # footwell width = 440 mm
        sole_z = 190.0 # sole height
        bench_w = 400.0 # benches span 220 to 620 mm
        bench_z = 460.0 # bench seat height (+270 mm above footwell!)
        coam_w = b - 180.0
        coam_z = sz + 45.0

        if x < 650:
            # Aft locker bench spanning across stern
            key_y = np.array([0.0, sole_w, sole_w + bench_w, coam_w, coam_w + 30.0, b])
            key_z = np.array([bench_z, bench_z, bench_z, coam_z, sz + 12.0, sz])
            z_deck_half = np.interp(y_deck_eval, key_y, key_z)
        else:
            # Main cockpit with footwell and side benches
            key_y = np.array([0.0, sole_w, sole_w + 15.0, sole_w + bench_w, coam_w, coam_w + 30.0, b])
            key_z = np.array([sole_z, sole_z, bench_z, bench_z, coam_z, sz + 12.0, sz])
            z_deck_half = np.interp(y_deck_eval, key_y, key_z)
    else:
        # Aft deck behind cockpit (x from 0 to 350)
        camber = 15.0 * (1.0 - (y_deck_eval / max(1.0, b))**2)
        z_deck_half = sz + camber

    # Mirror deck for full cross-section from +b (starboard) to -b (port)
    # y_deck_eval goes from 0 to +b. Reverse it: +b down to 0, then 0 down to -b
    y_deck_stbd = y_deck_eval[::-1] # from b down to 0
    z_deck_stbd = z_deck_half[::-1]
    y_deck_port = -y_deck_eval[1:]  # from -delta down to -b
    z_deck_port = z_deck_half[1:]

    y_deck_full = np.concatenate([y_deck_stbd, y_deck_port])
    z_deck_full = np.concatenate([z_deck_stbd, z_deck_port])

    # Port rub rail bead
    pts_rail_port_y = [-b, -b - rail_r, -b]
    pts_rail_port_z = [sz + rail_r*0.5, sz, sz - rail_r*0.7]

    # Port hull
    theta_port = np.linspace(np.pi/2 * 0.96, 0, n_hull_half)
    y_hull_port = -b * (np.sin(theta_port) ** p)
    z_hull_port = kz + (sz - kz) * (1.0 - np.cos(theta_port) ** q)

    # Assemble complete closed ring
    ring_y = np.concatenate([y_hull_stbd, pts_rail_stbd_y, y_deck_full[1:-1], pts_rail_port_y, y_hull_port[1:]])
    ring_z = np.concatenate([z_hull_stbd, pts_rail_stbd_z, z_deck_full[1:-1], pts_rail_port_z, z_hull_port[1:]])
    ring_x = np.full_like(ring_y, x)
    return np.column_stack([ring_x, ring_y, ring_z])

print("Generating ultra-smooth hull (101 stations)...")
nx = 101
x_vals = np.linspace(0, 5990, nx)
rings = [get_station_ring_v2(x) for x in x_vals]
m_pts = len(rings[0])
print(f"Mesh density: {nx} stations x {m_pts} points per ring = {nx * m_pts} vertices")

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
print(f"Smooth Hull Solid: Watertight={tm_hull.is_watertight}, Volume={tm_hull.volume:.1f} mm³")

# Render top deck view to inspect benches and hatch!
fig = plt.figure(figsize=(12, 6), dpi=150)
fig.patch.set_facecolor('#1a1e24')
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_facecolor('#1a1e24')

poly = Poly3DCollection(tm_hull.vertices[tm_hull.faces], alpha=0.95, edgecolor='#2b323c', linewidths=0.05)
poly.set_facecolor('#e2e8f0')
ax.add_collection3d(poly)
ax.set_xlim([0, 6000])
ax.set_ylim([-1300, 1300])
ax.set_zlim([-400, 1400])
ax.view_init(elev=75, azim=-90)
ax.axis('off')
plt.title("Updated Jouët Sheriff 600 - Deck & Cockpit Benches Top View", color='white', fontsize=14)
plt.tight_layout()
plt.savefig("test_deck_top_view.png", facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
plt.close()
print("Saved test_deck_top_view.png")
