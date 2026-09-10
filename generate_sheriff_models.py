import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import trimesh
import manifold3d as m3d
from scipy.interpolate import CubicSpline

def to_watertight_trimesh(manifold_obj):
    """Convert Manifold3D object to a verified watertight, correct-winding trimesh."""
    mesh_data = manifold_obj.to_mesh()
    tm = trimesh.Trimesh(
        vertices=mesh_data.vert_properties,
        faces=mesh_data.tri_verts,
        process=False
    )
    trimesh.repair.fix_winding(tm)
    trimesh.repair.fix_normals(tm)
    if tm.volume < 0:
        tm = trimesh.Trimesh(
            vertices=mesh_data.vert_properties,
            faces=mesh_data.tri_verts[:, [0, 2, 1]],
            process=False
        )
        trimesh.repair.fix_winding(tm)
        trimesh.repair.fix_normals(tm)
    return tm

def build_jouet_sheriff_v3():
    print("==========================================================")
    print("  JOUET SHERIFF 600 - MASTER HIGH-RES GENERATOR (V3)      ")
    print("  Incorporating Authentic Blueprint & Photo Details:      ")
    print("  - Full-beam coachroof (walkway on roof, no side deck)   ")
    print("  - Upper skylight slightly wrapping onto cabin roof      ")
    print("  - Side window contours on cabin sides                   ")
    print("  - Cockpit: tapered aft benches, forward rounded fillets ")
    print("  - Clean planar faces & straight line definitions        ")
    print("  - Flashforge Adventurer 3 (150x150x150 mm) Ready        ")
    print("==========================================================")

    # 1. Base Dimensions & Calibration from Bateaux Oct 1969 & Yachting France
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

        # Hull flare & deadrise exponents
        p = 1.25 + 0.55 * (t - 0.5) / 0.5 if t > 0.5 else 1.25 - 0.20 * (0.5 - t) / 0.5
        q = 1.35 - 0.25 * (t - 0.5) / 0.5 if t > 0.5 else 1.35 + 0.25 * (0.5 - t) / 0.5

        # 1. Starboard hull curve (keel to sheer)
        theta_s = np.linspace(0, np.pi/2 * 0.96, n_hull_half)
        ys_hull = b * (np.sin(theta_s) ** p)
        zs_hull = kz + (sz - kz) * (1.0 - np.cos(theta_s) ** q)

        # Rub rail bead at sheer
        rail_r = 22.0 if b > 50 else b * 0.15
        pts_rail_stbd_y = [b, b + rail_r, b]
        pts_rail_stbd_z = [sz - rail_r*0.7, sz, sz + rail_r*0.5]

        # 2. Deck & Superstructure cross-section across starboard half (y from 0 to b)
        y_deck_eval = np.linspace(0.0, b, n_deck_half)
        z_deck_half = np.zeros(n_deck_half)

        if x >= 4000:
            # Foredeck: pure smooth parabolic camber (sheds sea water)
            camber = 35.0 * (1.0 - (y_deck_eval / max(1.0, b))**2)
            z_deck_half = sz + camber

        elif x >= 2000:
            # Coachroof region (x = 2000 to 4000)
            # Full-beam coachroof: side deck is just the narrow gunwale / toe-rail margin.
            # Forward flat slope: exact planar transition from roof to foredeck between x=3100 and 3850
            if x >= 3100:
                slope_t = (x - 3100.0) / 750.0  # 0 at 3100, 1 at 3850
                roof_h = (1.0 - slope_t) * 380.0 + slope_t * 35.0
                cab_camber = (1.0 - slope_t) * 22.0 + slope_t * 35.0
                cab_top_half_w = (1.0 - slope_t) * (b * 0.78) + slope_t * (b * 0.35)
            else:
                roof_h = 380.0
                cab_camber = 22.0
                cab_top_half_w = b * 0.80

            roof_center_z = sz + roof_h + cab_camber
            roof_edge_z = sz + roof_h

            margin_w = max(18.0, b * 0.05)
            cab_base_w = b - margin_w

            # Exact planar bevel profile: Center roof -> Roof edge -> Cabin side wall -> Toe rail margin -> Sheer
            key_y = np.array([0.0, cab_top_half_w * 0.7, cab_top_half_w, cab_base_w, b])
            key_z = np.array([roof_center_z, roof_center_z - cab_camber*0.3, roof_edge_z, sz + 20.0, sz])
            z_deck_half = np.interp(y_deck_eval, key_y, key_z)

            # Forward Skylight / Window:
            # Sits high on forward slope (x=2950 to 3400) and slightly wraps onto cabin roof across x=3100!
            if 2950 <= x <= 3400:
                win_hw = 200.0 - 25.0 * (x - 2950.0) / 450.0
                for i, y in enumerate(y_deck_eval):
                    if y <= win_hw:
                        dist_to_edge = win_hw - y
                        if dist_to_edge < 22.0:
                            z_deck_half[i] += 16.0  # raised frame contour
                        else:
                            z_deck_half[i] += 10.0  # flat glass pane

            # Side Window Contours:
            # Distinct raised trapezoidal frame contour on outer cabin sides between x=2250 and 2850
            if 2250 <= x <= 2850:
                for i, y in enumerate(y_deck_eval):
                    if cab_top_half_w * 0.96 <= y <= cab_base_w * 0.98:
                        z_deck_half[i] += 14.0  # raised window frame contour

        elif x >= 250:
            # Cockpit region (x = 250 to 2000)
            sole_w = 220.0  # footwell half-width
            sole_z = 200.0  # flat footwell floor

            # Tapered benches: narrower at aft stern, wider at mid-cockpit
            if x < 650:
                bench_w = 280.0  # narrower aft bench
                bench_z = 470.0
            else:
                bench_w = 420.0  # full comfortable cockpit bench
                bench_z = 470.0

            # Forward rounded fillets near cabin entrance ("закругления")
            if x >= 1650:
                round_t = (x - 1650.0) / 350.0
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

        # Mirror for port side
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

        # Assemble full closed ring
        ring_y = np.concatenate([ys_hull, pts_rail_stbd_y, y_deck_full[1:-1], pts_rail_port_y, yp_hull[1:]])
        ring_z = np.concatenate([zs_hull, pts_rail_stbd_z, z_deck_full[1:-1], pts_rail_port_z, zp_hull[1:]])
        ring_x = np.full_like(ring_y, x)
        return np.column_stack([ring_x, ring_y, ring_z])

    # 121 stations along length for silky smooth surface
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

    # Solid Transom cap at x=0
    c_idx = len(all_verts)
    all_verts = np.vstack([all_verts, np.array([[0.0, 0.0, 300.0]])])
    for im in range(m_pts):
        im_next = (im + 1) % m_pts
        all_faces.append([c_idx, im_next, im])

    # Bow Stem cap at x=6000
    b_idx = len(all_verts)
    all_verts = np.vstack([all_verts, np.array([[6000.0, 0.0, 980.0]])])
    for im in range(m_pts):
        im_next = (im + 1) % m_pts
        all_faces.append([b_idx, (nx - 1) * m_pts + im, (nx - 1) * m_pts + im_next])

    hull_deck_mesh = trimesh.Trimesh(vertices=all_verts, faces=np.array(all_faces), process=False)
    trimesh.repair.fix_winding(hull_deck_mesh)
    trimesh.repair.fix_normals(hull_deck_mesh)
    print(f"Hull & Deck Solid: Watertight={hull_deck_mesh.is_watertight}")

    # 2. Authentic Cast-Iron Keel & Bulb
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

    # Mast Step Pad on Coachroof
    mast_pad = trimesh.creation.cylinder(radius=65.0, height=40.0)
    mast_pad.apply_translation([3200.0, 0.0, cs_sheer(3200) + 395.0])

    # Boolean Union via Manifold3D
    mh = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(hull_deck_mesh.vertices, dtype=np.float32), tri_verts=np.asarray(hull_deck_mesh.faces, dtype=np.uint32)))
    mk = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(keel_tm.vertices, dtype=np.float32), tri_verts=np.asarray(keel_tm.faces, dtype=np.uint32)))
    mp = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(mast_pad.vertices, dtype=np.float32), tri_verts=np.asarray(mast_pad.faces, dtype=np.uint32)))

    m_full = mh + mk + mp
    full_boat = to_watertight_trimesh(m_full)
    print(f"FULL BOAT V3 SOLID: Watertight={full_boat.is_watertight}, Volume={full_boat.volume:.1f} mm³")

    # 3. Waterline Model (DWL Z=0 cut with flat bottom)
    cut_box = trimesh.creation.box([14000.0, 6000.0, 2500.0])
    cut_box.apply_translation([3000.0, 0.0, -1250.0])
    m_cut_box = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(cut_box.vertices, dtype=np.float32), tri_verts=np.asarray(cut_box.faces, dtype=np.uint32)))
    m_wl = m_full - m_cut_box
    waterline_boat = to_watertight_trimesh(m_wl)
    print(f"WATERLINE SOLID: Watertight={waterline_boat.is_watertight}, Volume={waterline_boat.volume:.1f} mm³")

    # 4. Split Model: Deck (flat bottom) + Keel (flat top) with 2 registration sockets
    top_box = trimesh.creation.box([14000.0, 6000.0, 3000.0])
    top_box.apply_translation([3000.0, 0.0, 1500.0])
    m_top_box = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(top_box.vertices, dtype=np.float32), tri_verts=np.asarray(top_box.faces, dtype=np.uint32)))
    m_bottom = m_full - m_top_box

    # Alignment Sockets (отверстия под штифты / кусочки филамента 2 мм):
    pin_fwd = trimesh.creation.cylinder(radius=55.0, height=360.0)
    pin_fwd.apply_translation([3800.0, 0.0, 0.0])
    pin_aft = trimesh.creation.cylinder(radius=55.0, height=360.0)
    pin_aft.apply_translation([1500.0, 0.0, 0.0])

    m_pins = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(pin_fwd.vertices, dtype=np.float32), tri_verts=np.asarray(pin_fwd.faces, dtype=np.uint32)))
    m_pins = m_pins + m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(pin_aft.vertices, dtype=np.float32), tri_verts=np.asarray(pin_aft.faces, dtype=np.uint32)))

    # Both parts have a 100% FLAT surface at the waterline (Z=0) with sockets:
    m_deck_split = m_wl - m_pins
    m_keel_split = m_bottom - m_pins

    split_deck = to_watertight_trimesh(m_deck_split)
    split_keel = to_watertight_trimesh(m_keel_split)
    print(f"SPLIT DECK: Watertight={split_deck.is_watertight}")
    print(f"SPLIT KEEL: Watertight={split_keel.is_watertight}")

    # 5. Rudder & Tiller (Навесной руль с румпелем)
    # Mounted entirely OUTSIDE the transom (X < 0)!
    # Tiller attaches to top of rudder head (Z = 710 mm), passes ABOVE transom (Z=650),
    # and extends forward 850 mm horizontally at Z = 710 mm (above the cockpit benches!)
    r_blade = trimesh.creation.box([300.0, 50.0, 950.0])
    r_blade.apply_translation([-180.0, 0.0, -100.0])

    r_cheeks = trimesh.creation.box([120.0, 75.0, 360.0])
    r_cheeks.apply_translation([-60.0, 0.0, 500.0])

    # Tiller bar: solid wooden tiller extending forward above the cockpit
    r_tiller = trimesh.creation.box([850.0, 45.0, 45.0])
    r_tiller.apply_translation([350.0, 0.0, 710.0])

    # Transom mount pins
    p_top = trimesh.creation.cylinder(radius=25.0, height=80.0)
    rot_p = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
    p_top.apply_transform(rot_p)
    p_top.apply_translation([10.0, 0.0, 560.0])

    p_bot = trimesh.creation.cylinder(radius=25.0, height=80.0)
    p_bot.apply_transform(rot_p)
    p_bot.apply_translation([10.0, 0.0, 360.0])

    m_r = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(r_blade.vertices, dtype=np.float32), tri_verts=np.asarray(r_blade.faces, dtype=np.uint32)))
    m_rc = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(r_cheeks.vertices, dtype=np.float32), tri_verts=np.asarray(r_cheeks.faces, dtype=np.uint32)))
    m_rt = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(r_tiller.vertices, dtype=np.float32), tri_verts=np.asarray(r_tiller.faces, dtype=np.uint32)))
    m_pt = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(p_top.vertices, dtype=np.float32), tri_verts=np.asarray(p_top.faces, dtype=np.uint32)))
    m_pb = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(p_bot.vertices, dtype=np.float32), tri_verts=np.asarray(p_bot.faces, dtype=np.uint32)))
    m_rudder = m_r + m_rc + m_rt + m_pt + m_pb
    rudder_solid = to_watertight_trimesh(m_rudder)
    print(f"Rudder Solid: Watertight={rudder_solid.is_watertight}")

    # 6. Rigging (Мачта и гик)
    mast_cyl = trimesh.creation.cylinder(radius=45.0, height=6800.0)
    mast_cyl.apply_translation([0.0, 0.0, 3400.0])
    boom_cyl = trimesh.creation.cylinder(radius=35.0, height=2700.0)
    rot_b = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
    boom_cyl.apply_transform(rot_b)
    boom_cyl.apply_translation([-1350.0, 0.0, 600.0])
    spreader = trimesh.creation.box([40.0, 950.0, 30.0])
    spreader.apply_translation([0.0, 0.0, 3200.0])

    m_m = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(mast_cyl.vertices, dtype=np.float32), tri_verts=np.asarray(mast_cyl.faces, dtype=np.uint32)))
    m_b = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(boom_cyl.vertices, dtype=np.float32), tri_verts=np.asarray(boom_cyl.faces, dtype=np.uint32)))
    m_s = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(spreader.vertices, dtype=np.float32), tri_verts=np.asarray(spreader.faces, dtype=np.uint32)))
    m_rig = m_m + m_b + m_s
    rigging_solid = to_watertight_trimesh(m_rig)
    print(f"Rigging Solid: Watertight={rigging_solid.is_watertight}")

    # 7. Display Cradle (Кильблок-подставка)
    c_base = trimesh.creation.box([4500.0, 2200.0, 160.0])
    c_base.apply_translation([2600.0, 0.0, -930.0])
    p_fwd = trimesh.creation.box([450.0, 1900.0, 750.0])
    p_fwd.apply_translation([3800.0, 0.0, -560.0])
    p_aft = trimesh.creation.box([450.0, 1900.0, 750.0])
    p_aft.apply_translation([1400.0, 0.0, -560.0])

    m_c_base = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(c_base.vertices, dtype=np.float32), tri_verts=np.asarray(c_base.faces, dtype=np.uint32)))
    m_p_fwd = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(p_fwd.vertices, dtype=np.float32), tri_verts=np.asarray(p_fwd.faces, dtype=np.uint32)))
    m_p_aft = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(p_aft.vertices, dtype=np.float32), tri_verts=np.asarray(p_aft.faces, dtype=np.uint32)))
    m_cradle_raw = m_c_base + m_p_fwd + m_p_aft

    cutter = full_boat.copy()
    cutter.apply_scale(1.03)
    m_cutter = m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(cutter.vertices, dtype=np.float32), tri_verts=np.asarray(cutter.faces, dtype=np.uint32)))
    m_cradle = m_cradle_raw - m_cutter
    cradle_solid = to_watertight_trimesh(m_cradle)
    print(f"Display Cradle Solid: Watertight={cradle_solid.is_watertight}")

    # -------------------------------------------------------------
    # 8. EXPORT SUITE (Scale 1:45 for Adventurer 3 & 1:1)
    # -------------------------------------------------------------
    os.makedirs("stl_output", exist_ok=True)
    scale_1to45 = 1.0 / 45.0

    models = [
        ("Jouet_Sheriff_Waterline_1to45.stl", waterline_boat, scale_1to45, "Ватерлинейная модель (печать БЕЗ поддержек, плоское дно)"),
        ("Jouet_Sheriff_Full_1to45.stl", full_boat, scale_1to45, "Монолитный корпус с килем, рубкой и окнами"),
        ("Jouet_Sheriff_Split_Deck_1to45.stl", split_deck, scale_1to45, "Верхняя половина (палуба со скамейками, плоский низ со стыковочными пазами)"),
        ("Jouet_Sheriff_Split_Keel_1to45.stl", split_keel, scale_1to45, "Нижняя половина (киль и днище, плоский верх со стыковочными штифтами)"),
        ("Jouet_Sheriff_Rudder_1to45.stl", rudder_solid, scale_1to45, "Навесной руль с румпелем (монтируется снаружи транца)"),
        ("Jouet_Sheriff_Rigging_1to45.stl", rigging_solid, scale_1to45, "Мачта с гиком и краспицами (печать по диагонали стола 212 мм)"),
        ("Jouet_Sheriff_Display_Cradle_1to45.stl", cradle_solid, scale_1to45, "Настольный кильблок-подставка (100x49x18 мм)"),
        ("Jouet_Sheriff_Full_1to1.stl", full_boat, 1.0, "Полный корпус 1:1 (LOA 6000 мм)"),
        ("Jouet_Sheriff_Waterline_1to1.stl", waterline_boat, 1.0, "Ватерлинейный корпус 1:1"),
    ]

    print("\n----------------------------------------------------------")
    print("  EXPORTING ALL STL FILES (V3)...")
    print("----------------------------------------------------------")
    for fname, m_obj, sc, desc in models:
        out = m_obj.copy()
        if sc != 1.0:
            out.apply_scale(sc)
        p = os.path.join("stl_output", fname)
        out.export(p)
        b = out.extents
        print(f"[{fname}]")
        print(f"  Назначение:  {desc}")
        print(f"  Размеры:     X={b[0]:.1f} мм, Y={b[1]:.1f} мм, Z={b[2]:.1f} мм")
        print(f"  Watertight:  {out.is_watertight} (Solid manifold), Полигонов: {len(out.faces)}")
        print()

    # Save 3D assembly scene OBJ with properly positioned rudder outside transom
    sc_boat = full_boat.copy()
    sc_boat.apply_scale(scale_1to45)
    sc_cradle = cradle_solid.copy()
    sc_cradle.apply_scale(scale_1to45)
    sc_rudder = rudder_solid.copy()
    sc_rudder.apply_scale(scale_1to45)

    scene = trimesh.Scene([sc_boat, sc_cradle, sc_rudder])
    scene.export("stl_output/Jouet_Sheriff_Assembly_Preview.obj")
    print("Exported updated 3D preview scene: stl_output/Jouet_Sheriff_Assembly_Preview.obj")

if __name__ == "__main__":
    build_jouet_sheriff_v3()
