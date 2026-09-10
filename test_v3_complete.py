import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import trimesh
import manifold3d as m3d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Calibration Splines
stations_x = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000], dtype=float)
sheer_z_pts = np.array([650, 675, 705, 730, 748, 760, 780, 810, 850, 895, 935, 960, 980], dtype=float)
beam_pts = np.array([800, 930, 1040, 1115, 1148, 1155, 1142, 1085, 975, 780, 510, 240, 0], dtype=float)
keel_z_pts = np.array([-50, -120, -195, -255, -295, -320, -310, -270, -200, -110, -15, 350, 980], dtype=float)

cs_sheer = CubicSpline(stations_x, sheer_z_pts)
cs_beam = CubicSpline(stations_x, beam_pts)
cs_keel = CubicSpline(stations_x, keel_z_pts)

def get_v3_ring(x, n_hull_half=36, n_deck_half=41):
    b = max(4.0, float(cs_beam(x)))
    kz = float(cs_keel(x))
    sz = float(cs_sheer(x))
    t = x / 6000.0

    # 1. Hull deadrise & flare parameters
    p = 1.25 + 0.55 * (t - 0.5) / 0.5 if t > 0.5 else 1.25 - 0.20 * (0.5 - t) / 0.5
    q = 1.35 - 0.25 * (t - 0.5) / 0.5 if t > 0.5 else 1.35 + 0.25 * (0.5 - t) / 0.5

    # Starboard hull curve (keel to sheer)
    theta_s = np.linspace(0, np.pi/2 * 0.96, n_hull_half)
    ys_hull = b * (np.sin(theta_s) ** p)
    zs_hull = kz + (sz - kz) * (1.0 - np.cos(theta_s) ** q)

    # Crisp Rub Rail bead at sheer
    rail_r = 22.0 if b > 50 else b * 0.15
    pts_rail_stbd_y = [b, b + rail_r, b]
    pts_rail_stbd_z = [sz - rail_r*0.7, sz, sz + rail_r*0.5]

    # 2. Deck across from centerline 0 to sheer +b
    y_deck_eval = np.linspace(0.0, b, n_deck_half)
    z_deck_half = np.zeros(n_deck_half)

    if x >= 3700:
        # Foredeck: pure clean camber
        camber = 35.0 * (1.0 - (y_deck_eval / max(1.0, b))**2)
        z_deck_half = sz + camber

    elif x >= 2000:
        # Full-beam coachroof (no side passages, passage is on the roof!)
        # Forward slope: from x=3100 to 3700 (perfect flat slope!)
        if x >= 3100:
            slope_t = (x - 3100.0) / 600.0 # 0 at 3100, 1 at 3700
            roof_h = (1.0 - slope_t) * 380.0 + slope_t * 35.0
            cab_camber = (1.0 - slope_t) * 22.0 + slope_t * 35.0
        else:
            roof_h = 380.0
            cab_camber = 22.0

        roof_center_z = sz + roof_h + cab_camber
        roof_edge_z = sz + roof_h

        # The cabin side rises almost directly from the sheer (25 mm toe-rail margin)
        toe_rail_w = max(15.0, b * 0.04)
        cab_w = b - toe_rail_w

        key_y = np.array([0.0, cab_w * 0.85, cab_w, b - 5.0, b])
        key_z = np.array([roof_center_z, roof_edge_z, sz + 25.0, sz + 15.0, sz])
        z_deck_half = np.interp(y_deck_eval, key_y, key_z)

        # Forward Window / Skylight:
        # "Окно должно слегка заходить на крышу рубки, а не на нос."
        # Located high on the forward slope: x in 2950..3450 (wraps roof edge at x=3100!)
        if 2950 <= x <= 3450:
            # Width: 420 mm (half-width 210 mm)
            hatch_w = 210.0 - 20.0 * (x - 2950) / 500.0
            for i, y in enumerate(y_deck_eval):
                if y <= hatch_w:
                    edge_dist = hatch_w - y
                    if edge_dist < 22.0:
                        z_deck_half[i] += 18.0 # raised frame
                    else:
                        z_deck_half[i] += 12.0 # flat glass pane

        # Side Windows:
        # "Сбоку должны быть контуры окон."
        # Located on cabin sides between x=2300 and 2850
        if 2300 <= x <= 2850:
            # Window contour is on the outer cabin wall (y near cab_w)
            for i, y in enumerate(y_deck_eval):
                if cab_w * 0.88 <= y <= cab_w * 0.98:
                    z_deck_half[i] += 14.0 # raised trapezoidal frame

    elif x >= 250:
        # Cockpit region (x in 250..2000)
        # Features:
        # - Narrower benches near transom (x=250..650)
        # - Wider benches at mid-cockpit (x=650..1700)
        # - Rounded smooth coaming fillets near cabin entrance (x=1700..2000)
        sole_w = 220.0
        sole_z = 200.0 # flat footwell floor

        # Aft bench vs forward bench width:
        if x < 650:
            bench_w = 300.0 # narrower aft bench!
            bench_z = 470.0
        else:
            bench_w = 420.0 # standard comfortable seat
            bench_z = 470.0

        # Coaming fillet / curve near cabin entrance:
        if x >= 1700:
            # Rounded corner transition ("закругления")
            round_t = (x - 1700.0) / 300.0 # 0 at 1700, 1 at 2000
            # Coaming sweeps inward towards companionway entrance
            coam_w = (1.0 - round_t) * (b - 160.0) + round_t * (sole_w + 140.0)
            coam_z = sz + 45.0 + round_t * 120.0 # rises to meet cabin bulkhead
        else:
            coam_w = b - 160.0
            coam_z = sz + 45.0

        # Symmetrical cockpit profile
        if x < 550:
            # Aft stern locker bench
            key_y = np.array([0.0, sole_w, sole_w + bench_w, coam_w, b])
            key_z = np.array([bench_z, bench_z, bench_z, coam_z, sz])
            z_deck_half = np.interp(y_deck_eval, key_y, key_z)
        else:
            # Cockpit seats and sunken footwell
            key_y = np.array([0.0, sole_w, sole_w + 12.0, sole_w + bench_w, coam_w, b])
            key_z = np.array([sole_z, sole_z, bench_z, bench_z, coam_z, sz])
            z_deck_half = np.interp(y_deck_eval, key_y, key_z)

    else:
        # Aft deck behind cockpit (x from 0 to 250)
        camber = 12.0 * (1.0 - (y_deck_eval / max(1.0, b))**2)
        z_deck_half = sz + camber

    # Mirror deck to create full symmetrical section from +b to -b
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
    theta_p = np.linspace(np.pi/2 * 0.96, 0, n_hull_half)
    yp_hull = -b * (np.sin(theta_p) ** p)
    zp_hull = kz + (sz - kz) * (1.0 - np.cos(theta_p) ** q)

    # Full manifold ring
    ring_y = np.concatenate([ys_hull, pts_rail_stbd_y, y_deck_full[1:-1], pts_rail_port_y, yp_hull[1:]])
    ring_z = np.concatenate([zs_hull, pts_rail_stbd_z, z_deck_full[1:-1], pts_rail_port_z, zp_hull[1:]])
    ring_x = np.full_like(ring_y, x)
    return np.column_stack([ring_x, ring_y, ring_z])

print("Testing V3 ring generation with 101 stations...")
nx = 101
x_vals = np.linspace(0, 5990, nx)
rings = [get_v3_ring(x) for x in x_vals]
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

# Solid Transom
c_idx = len(all_verts)
all_verts = np.vstack([all_verts, np.array([[0.0, 0.0, 300.0]])])
for im in range(m_pts):
    im_next = (im + 1) % m_pts
    all_faces.append([c_idx, im_next, im])

# Bow Stem
b_idx = len(all_verts)
all_verts = np.vstack([all_verts, np.array([[6000.0, 0.0, 980.0]])])
for im in range(m_pts):
    im_next = (im + 1) % m_pts
    all_faces.append([b_idx, (nx - 1) * m_pts + im, (nx - 1) * m_pts + im_next])

tm_v3 = trimesh.Trimesh(vertices=all_verts, faces=np.array(all_faces), process=False)
trimesh.repair.fix_winding(tm_v3)
trimesh.repair.fix_normals(tm_v3)
print(f"V3 Hull Solid: Watertight={tm_v3.is_watertight}, Volume={tm_v3.volume:.1f} mm³")

# Keel Solid
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

# Mast Pad
mast_pad = trimesh.creation.cylinder(radius=65.0, height=45.0)
mast_pad.apply_translation([3200.0, 0.0, cs_sheer(3200) + 390.0])

# Union
mh = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(tm_v3.vertices, dtype=np.float32), tri_verts=np.asarray(tm_v3.faces, dtype=np.uint32)))
mk = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(keel_tm.vertices, dtype=np.float32), tri_verts=np.asarray(keel_tm.faces, dtype=np.uint32)))
mp = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(mast_pad.vertices, dtype=np.float32), tri_verts=np.asarray(mast_pad.faces, dtype=np.uint32)))

m_full = mh + mk + mp
mesh_full = m_full.to_mesh()
full_boat_v3 = trimesh.Trimesh(vertices=mesh_full.vert_properties, faces=mesh_full.tri_verts, process=False)
trimesh.repair.fix_winding(full_boat_v3)
trimesh.repair.fix_normals(full_boat_v3)
print(f"FULL BOAT V3 SOLID: Watertight={full_boat_v3.is_watertight}, Volume={full_boat_v3.volume:.1f} mm³")

# Render comparison views
fig = plt.figure(figsize=(16, 10), dpi=150)
fig.patch.set_facecolor('#1a1e24')

# View 1: Top Deck Plan
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.set_facecolor('#1a1e24')
poly1 = Poly3DCollection(full_boat_v3.vertices[full_boat_v3.faces], alpha=0.95, edgecolor='#2b323c', linewidths=0.05)
poly1.set_facecolor('#e2e8f0')
ax1.add_collection3d(poly1)
ax1.set_xlim([0, 6000])
ax1.set_ylim([-1300, 1300])
ax1.set_zlim([-400, 1400])
ax1.view_init(elev=82, azim=-90)
ax1.axis('off')
ax1.set_title("V3 Top Deck Plan: Full-Beam Cabin, Front Window, Tapered Aft Benches", color='white')

# View 2: Forward 3/4 View
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.set_facecolor('#1a1e24')
poly2 = Poly3DCollection(full_boat_v3.vertices[full_boat_v3.faces], alpha=0.95, edgecolor='#2b323c', linewidths=0.05)
poly2.set_facecolor('#e2e8f0')
ax2.add_collection3d(poly2)
ax2.set_xlim([0, 6000])
ax2.set_ylim([-1300, 1300])
ax2.set_zlim([-400, 1400])
ax2.view_init(elev=20, azim=-35)
ax2.axis('off')
ax2.set_title("V3 Front 3/4: Flat Planar Slope & Upper Window, Side Windows", color='white')

# View 3: Aft Cockpit View
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.set_facecolor('#1a1e24')
poly3 = Poly3DCollection(full_boat_v3.vertices[full_boat_v3.faces], alpha=0.95, edgecolor='#2b323c', linewidths=0.05)
poly3.set_facecolor('#e2e8f0')
ax3.add_collection3d(poly3)
ax3.set_xlim([0, 6000])
ax3.set_ylim([-1300, 1300])
ax3.set_zlim([-400, 1400])
ax3.view_init(elev=25, azim=-140)
ax3.axis('off')
ax3.set_title("V3 Aft Cockpit: Benches, Forward Rounded Fillets & Sunken Floor", color='white')

# View 4: Side Elevation View
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.set_facecolor('#1a1e24')
poly4 = Poly3DCollection(full_boat_v3.vertices[full_boat_v3.faces], alpha=0.95, edgecolor='#2b323c', linewidths=0.05)
poly4.set_facecolor('#e2e8f0')
ax4.add_collection3d(poly4)
ax4.set_xlim([0, 6000])
ax4.set_ylim([-1300, 1300])
ax4.set_zlim([-400, 1400])
ax4.view_init(elev=0, azim=-90)
ax4.axis('off')
ax4.set_title("V3 Side Profile: Pure Lines Plan Match, Flush Cabin Sides", color='white')

plt.tight_layout()
plt.savefig("test_v3_renders.png", facecolor=fig.get_facecolor(), edgecolor='none', dpi=150)
plt.close()
print("Saved test_v3_renders.png")
