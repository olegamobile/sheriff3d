import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import trimesh
import manifold3d as m3d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -------------------------------------------------------------
# JOUËT SHERIFF 600 - EXACT OFFSETS & BLUEPRINT CALIBRATION
# -------------------------------------------------------------
stations_x = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000], dtype=float)
sheer_z_pts = np.array([650, 675, 705, 730, 748, 760, 780, 810, 850, 895, 935, 960, 980], dtype=float)
beam_pts = np.array([800, 930, 1040, 1115, 1148, 1155, 1142, 1085, 975, 780, 510, 240, 0], dtype=float)
keel_z_pts = np.array([-50, -120, -195, -255, -295, -320, -310, -270, -200, -110, -15, 350, 980], dtype=float)

cs_sheer = CubicSpline(stations_x, sheer_z_pts)
cs_beam = CubicSpline(stations_x, beam_pts)
cs_keel = CubicSpline(stations_x, keel_z_pts)

def get_sheriff_ring(x, n_hull_half=36, n_deck_half=45):
    b = max(4.0, float(cs_beam(x)))
    kz = float(cs_keel(x))
    sz = float(cs_sheer(x))
    t = x / 6000.0

    # 1. Hull Cross Section (Superellipse parameterized by station)
    p = 1.25 + 0.55 * (t - 0.5) / 0.5 if t > 0.5 else 1.25 - 0.20 * (0.5 - t) / 0.5
    q = 1.35 - 0.25 * (t - 0.5) / 0.5 if t > 0.5 else 1.35 + 0.25 * (0.5 - t) / 0.5

    theta_s = np.linspace(0, np.pi/2 * 0.96, n_hull_half)
    ys_hull = b * (np.sin(theta_s) ** p)
    zs_hull = kz + (sz - kz) * (1.0 - np.cos(theta_s) ** q)

    # Crisp Rubrail Bead
    rail_r = 22.0 if b > 50 else b * 0.15
    pts_rail_stbd_y = [b, b + rail_r, b]
    pts_rail_stbd_z = [sz - rail_r*0.7, sz, sz + rail_r*0.5]

    # 2. Deck Cross Section across starboard half: y from 0 to b
    y_deck_eval = np.linspace(0.0, b, n_deck_half)
    z_deck_half = np.zeros(n_deck_half)

    if x >= 4000:
        # Foredeck: pure clean camber (curved deck to shed water)
        camber = 35.0 * (1.0 - (y_deck_eval / max(1.0, b))**2)
        z_deck_half = sz + camber

    elif x >= 2000:
        # Coachroof region (x = 2000 to 4000)
        # Features:
        # - Full beam coachroof (no sunken side passages; walkway is on roof)
        # - Tapered forward nose: starts at x=3100 and tapers down to foredeck at x=3850
        # - Forward flat slope: exact linear plane
        if x >= 3100:
            slope_t = (x - 3100.0) / 750.0  # 0 at 3100, 1 at 3850
            roof_h = (1.0 - slope_t) * 380.0 + slope_t * 35.0
            cab_camber = (1.0 - slope_t) * 22.0 + slope_t * 35.0
            # Forward chamfer width narrows forward
            cab_top_half_w = (1.0 - slope_t) * (b * 0.78) + slope_t * (b * 0.35)
        else:
            roof_h = 380.0
            cab_camber = 22.0
            cab_top_half_w = b * 0.80

        roof_center_z = sz + roof_h + cab_camber
        roof_edge_z = sz + roof_h

        # The cabin side meets the deck margin right near the sheer (toe rail margin)
        margin_w = max(18.0, b * 0.05)
        cab_base_w = b - margin_w

        # Piecewise exact linear segments: Center roof -> Roof edge -> Cabin side bevel -> Toe rail -> Sheer
        key_y = np.array([0.0, cab_top_half_w * 0.7, cab_top_half_w, cab_base_w, b])
        key_z = np.array([roof_center_z, roof_center_z - cab_camber*0.3, roof_edge_z, sz + 20.0, sz])
        z_deck_half = np.interp(y_deck_eval, key_y, key_z)

        # Forward Window / Skylight:
        # "Окно должно слегка заходить на крышу рубки, а не на нос."
        # Centered on forward slope at x from 2950 to 3400 (crossing the roof break at 3100!)
        if 2950 <= x <= 3400:
            win_hw = 200.0 - 25.0 * (x - 2950.0) / 450.0
            for i, y in enumerate(y_deck_eval):
                if y <= win_hw:
                    # Raised frame with sharp edge
                    dist_to_edge = win_hw - y
                    if dist_to_edge < 22.0:
                        z_deck_half[i] += 16.0  # frame
                    else:
                        z_deck_half[i] += 10.0  # glass pane

        # Side Windows:
        # "Сбоку должны быть контуры окон."
        # Distinct trapezoidal raised frame on cabin sides between x=2250 and 2850
        if 2250 <= x <= 2850:
            for i, y in enumerate(y_deck_eval):
                if cab_top_half_w * 0.96 <= y <= cab_base_w * 0.98:
                    z_deck_half[i] += 14.0  # window contour

    elif x >= 250:
        # Cockpit region (x = 250 to 2000)
        # Features:
        # 1. Narrower benches at the aft end (x=250..650)
        # 2. Wider seats in the mid-cockpit (x=650..1700)
        # 3. Smooth rounded coamings near cabin entrance ("закругления", x=1700..2000)
        # 4. Sunken cockpit sole (floor)
        sole_w = 220.0
        sole_z = 200.0  # flat footwell floor

        if x < 650:
            # Narrower benches at the stern
            bench_w = 280.0
            bench_z = 470.0
        else:
            # Full comfortable seating benches
            bench_w = 420.0
            bench_z = 470.0

        if x >= 1650:
            # Rounded transition towards companionway ("закругления")
            round_t = (x - 1650.0) / 350.0
            # Coaming curves inward and sweeps up to the cabin bulkhead
            coam_w = (1.0 - round_t) * (b - 150.0) + round_t * (sole_w + 120.0)
            coam_z = sz + 40.0 + round_t * 140.0
        else:
            coam_w = b - 150.0
            coam_z = sz + 40.0

        if x < 500:
            # Aft stern locker bench
            key_y = np.array([0.0, sole_w, sole_w + bench_w, coam_w, b])
            key_z = np.array([bench_z, bench_z, bench_z, coam_z, sz])
        else:
            # Sunken footwell + bench seat
            key_y = np.array([0.0, sole_w, sole_w + 10.0, sole_w + bench_w, coam_w, b])
            key_z = np.array([sole_z, sole_z, bench_z, bench_z, coam_z, sz])
        z_deck_half = np.interp(y_deck_eval, key_y, key_z)

    else:
        # Aft deck behind cockpit (x = 0 to 250)
        camber = 12.0 * (1.0 - (y_deck_eval / max(1.0, b))**2)
        z_deck_half = sz + camber

    # Mirror deck for port side
    y_deck_stbd = y_deck_eval[::-1]
    z_deck_stbd = z_deck_half[::-1]
    y_deck_port = -y_deck_eval[1:]
    z_deck_port = z_deck_half[1:]

    y_deck_full = np.concatenate([y_deck_stbd, y_deck_port])
    z_deck_full = np.concatenate([z_deck_stbd, z_deck_port])

    # Port rub rail bead
    pts_rail_port_y = [-b, -b - rail_r, -b]
    pts_rail_port_z = [sz + rail_r*0.5, sz, sz - rail_r*0.7]

    # Port hull curve
    theta_p = np.linspace(np.pi/2 * 0.96, 0, n_hull_half)
    yp_hull = -b * (np.sin(theta_p) ** p)
    zp_hull = kz + (sz - kz) * (1.0 - np.cos(theta_p) ** q)

    # Construct closed manifold loop
    ring_y = np.concatenate([ys_hull, pts_rail_stbd_y, y_deck_full[1:-1], pts_rail_port_y, yp_hull[1:]])
    ring_z = np.concatenate([zs_hull, pts_rail_stbd_z, z_deck_full[1:-1], pts_rail_port_z, zp_hull[1:]])
    ring_x = np.full_like(ring_y, x)
    return np.column_stack([ring_x, ring_y, ring_z])

print("Generating high-resolution Sheriff 600 solid...")
nx = 121
x_vals = np.linspace(0, 5990, nx)
rings = [get_sheriff_ring(x) for x in x_vals]
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

# Solid Transom cap
c_idx = len(all_verts)
all_verts = np.vstack([all_verts, np.array([[0.0, 0.0, 300.0]])])
for im in range(m_pts):
    im_next = (im + 1) % m_pts
    all_faces.append([c_idx, im_next, im])

# Bow Stem cap
b_idx = len(all_verts)
all_verts = np.vstack([all_verts, np.array([[6000.0, 0.0, 980.0]])])
for im in range(m_pts):
    im_next = (im + 1) % m_pts
    all_faces.append([b_idx, (nx - 1) * m_pts + im, (nx - 1) * m_pts + im_next])

tm_hull = trimesh.Trimesh(vertices=all_verts, faces=np.array(all_faces), process=False)
trimesh.repair.fix_winding(tm_hull)
trimesh.repair.fix_normals(tm_hull)
print(f"Hull Watertight: {tm_hull.is_watertight}")

# Authentic Cast-Iron Keel
keel_stations_z = np.linspace(-260, -850, 21)
keel_verts = []
keel_faces = []
keel_grid = np.zeros((21, 28), dtype=int)
k_count = 0
for iz, z in enumerate(keel_stations_z):
    frac = (z - (-260)) / (-850 - (-260))
    x_le = 3050 - frac * 100
    x_te = 2050 + frac * 200
    chord = x_le - x_te
    max_thick = (95.0 + ((frac - 0.85)/0.15)*90.0) if frac > 0.85 else (95.0 * (1.0 - 0.2*frac))
    angles = np.linspace(0, 2*np.pi, 28, endpoint=False)
    for ia, ang in enumerate(angles):
        xc = 0.5 * (1.0 - np.cos(ang))
        yt = 5.0 * (0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*(xc**2) + 0.2843*(xc**3) - 0.1015*(xc**4))
        px = x_te + xc * chord
        py = np.sin(ang) * yt * max_thick
        keel_verts.append([px, py, z])
        keel_grid[iz, ia] = k_count
        k_count += 1

for iz in range(20):
    for ia in range(28):
        v0 = keel_grid[iz, ia]
        v1 = keel_grid[iz + 1, ia]
        v2 = keel_grid[iz + 1, (ia + 1) % 28]
        v3 = keel_grid[iz, (ia + 1) % 28]
        keel_faces.append([v0, v1, v2])
        keel_faces.append([v0, v2, v3])

bulb_bottom = k_count
keel_verts.append([2600.0, 0.0, -855.0])
k_count += 1
for ia in range(28):
    keel_faces.append([bulb_bottom, keel_grid[20, (ia + 1) % 28], keel_grid[20, ia]])

keel_top = k_count
keel_verts.append([2550.0, 0.0, -250.0])
k_count += 1
for ia in range(28):
    keel_faces.append([keel_top, keel_grid[0, ia], keel_grid[0, (ia + 1) % 28]])

keel_tm = trimesh.Trimesh(vertices=np.array(keel_verts), faces=np.array(keel_faces), process=False)
trimesh.repair.fix_winding(keel_tm)
trimesh.repair.fix_normals(keel_tm)

# Mast Step Pad
mast_pad = trimesh.creation.cylinder(radius=65.0, height=40.0)
mast_pad.apply_translation([3200.0, 0.0, cs_sheer(3200) + 395.0])

# Boolean Union via Manifold3D
mh = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(tm_hull.vertices, dtype=np.float32), tri_verts=np.asarray(tm_hull.faces, dtype=np.uint32)))
mk = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(keel_tm.vertices, dtype=np.float32), tri_verts=np.asarray(keel_tm.faces, dtype=np.uint32)))
mp = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(mast_pad.vertices, dtype=np.float32), tri_verts=np.asarray(mast_pad.faces, dtype=np.uint32)))

m_full = mh + mk + mp
mesh_full = m_full.to_mesh()
full_boat = trimesh.Trimesh(vertices=mesh_full.vert_properties, faces=mesh_full.tri_verts, process=False)
trimesh.repair.fix_winding(full_boat)
trimesh.repair.fix_normals(full_boat)
print(f"Full Solid: Watertight={full_boat.is_watertight}, Volume={full_boat.volume:.1f} mm³")

# Save as test preview
full_boat.export("test_full_v3.stl")
print("Saved test_full_v3.stl")
