import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import trimesh
import manifold3d as m3d
from scipy.interpolate import CubicSpline

# ============================================================================
#  DECK / SUPERSTRUCTURE PARAMETERS (all in 1:1 mm, X=0 transom, X=6000 stem,
#  Z=0 design waterline). Hull lines below the sheer are NOT affected by these.
# ============================================================================
DECK_CAMBER      = 35.0     # crown of the flat deck (foredeck, gunwale strips)

# Coachroof
X_BULKHEAD       = 2200.0   # aft cabin bulkhead at deck level
X_ROOF_FRONT     = 3400.0   # roof / forward-slope crease (mast stands just aft of it)
CABIN_SIDE_H     = 340.0    # cabin side wall height above the deck at the wall base
CABIN_ROOF_CAMBER = 25.0    # roof crown
CABIN_MARGIN     = 90.0     # narrow side-deck strip between sheer and cabin wall
WALL_TUMBLEHOME  = np.radians(20.0)   # cabin walls lean inward
SLOPE_ANGLE      = np.radians(25.0)   # forward slope of the coachroof
BULKHEAD_RAKE    = np.radians(8.0)    # aft bulkhead leans forward at the top
ROOF_AFT_R       = 150.0    # plan-view radius of the aft roof corners
# Forward of X_ROOF_FRONT the roof/slope outline is a quarter-ellipse in plan view (rounded nose);
# the side walls keep following the deck edge and flare out to meet it.

# Low toe rails ("стеночки") along the foredeck edges, continuing the cabin walls to the bow
RAIL_T           = 45.0
RAIL_H_AFT       = 130.0
RAIL_H_FWD       = 60.0
RAIL_X1          = 5800.0

# Forward hatch (tinted acrylic, hinged, no frame) on the forward slope
HATCH_X0, HATCH_X1 = 3440.0, 3860.0
HATCH_HALF_W     = 300.0
HATCH_RAISE      = 15.0

# Side windows (trapezoid, aluminium bezel, tinted glass) — nearly full wall height
WIN_X_BOT        = (2400.0, 3100.0)
WIN_X_TOP        = (2540.0, 2960.0)
WIN_Z_ABOVE_BASE = 45.0
WIN_H            = 250.0
WIN_FRAME_W      = 35.0
WIN_FRAME_RAISE  = 12.0
WIN_GLASS_RECESS = 8.0

# Companionway door (closed, recessed outline in the aft bulkhead)
DOOR_HALF_W      = 280.0
DOOR_Z0, DOOR_Z1 = 550.0, 1030.0
DOOR_RECESS      = 12.0

# Mast step pad
MAST_X           = 3330.0
MAST_PAD_R       = 70.0

# Cockpit: benches and footwell run from the transom wall to the bulkhead (no aft seat)
GUNWALE_W        = 170.0    # flat strip along the sheer, its inner face is the coaming
X_COCKPIT_AFT    = 80.0     # transom wall thickness
SEAT_Z           = 500.0    # horizontal seat plane
FLOOR_Z          = 230.0    # horizontal footwell sole (must stay above the deck pin sockets)
FOOTWELL_HW_AFT  = 330.0    # footwell half-width at the transom
FOOTWELL_HW_FWD  = 500.0    # footwell half-width at the bulkhead
SEAT_END_R       = 450.0    # benches end at the bulkhead in a quarter-round shoulder at deck height

# Foot-brace board on two posts + transverse boom-vang beam in the footwell
POST_X           = (1050.0, 1650.0)
POST_W           = 60.0
BOARD_T          = 40.0
BOARD_Z          = (350.0, 470.0)
BEAM_X           = 1650.0
BEAM_W           = 60.0
BEAM_Z           = (420.0, 480.0)

# Mooring cleat, athwartships, in the middle of the foredeck
CLEAT_X          = 5000.0

# Alignment pins: keel socket depth stays 180 (already printed); deck socket is shallower
PIN_R            = 55.0
KEEL_SOCKET_DEPTH = 180.0
DECK_SOCKET_DEPTH = 130.0


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


def to_manifold(tm):
    tm = tm.copy()
    trimesh.repair.fix_winding(tm)
    trimesh.repair.fix_normals(tm)
    if tm.volume < 0:
        tm.invert()
    m = m3d.Manifold(m3d.Mesh(
        vert_properties=np.asarray(tm.vertices, dtype=np.float32),
        tri_verts=np.asarray(tm.faces, dtype=np.uint32)))
    if m.is_empty():
        raise RuntimeError("Manifold construction failed (non-manifold input mesh)")
    return m


def mbox(x0, x1, y0, y1, z0, z1):
    b = trimesh.creation.box([x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    return to_manifold(b)


def loft(sections):
    """Closed solid from a list of closed polygon rings (same point count, same order)
    with fan caps from the ring centroid at both ends."""
    sections = [np.asarray(s, dtype=float) for s in sections]
    n = len(sections[0])
    verts = np.concatenate(sections, axis=0)
    faces = []
    for i in range(len(sections) - 1):
        a, b = i * n, (i + 1) * n
        for j in range(n):
            k = (j + 1) % n
            faces.append([a + j, b + j, b + k])
            faces.append([a + j, b + k, a + k])
    c0 = len(verts)
    verts = np.vstack([verts, sections[0].mean(axis=0)])
    for j in range(n):
        faces.append([c0, (j + 1) % n, j])
    c1 = len(verts)
    verts = np.vstack([verts, sections[-1].mean(axis=0)])
    base = (len(sections) - 1) * n
    for j in range(n):
        faces.append([c1, base + j, base + (j + 1) % n])
    return to_manifold(trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False))


def build_jouet_sheriff_v4():
    print("==========================================================")
    print("  JOUET SHERIFF 600 - MASTER GENERATOR (V4, photo-based)  ")
    print("  Hull/keel unchanged. Deck rebuilt via planar CSG:       ")
    print("  - long flat foredeck with mooring cleat                 ")
    print("  - full-beam coachroof, 25 deg planar slope, acrylic hatch")
    print("  - trapezoid side windows, raked bulkhead, door outline   ")
    print("  - open cockpit: gunwale strips, benches, aft seat, sole  ")
    print("  - locker box, foot-brace board on posts, vang beam       ")
    print("==========================================================")

    # 1. Base Dimensions & Calibration from Bateaux Oct 1969 & Yachting France
    stations_x = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000], dtype=float)
    sheer_z_pts = np.array([650, 675, 705, 730, 748, 760, 780, 810, 850, 895, 935, 960, 980], dtype=float)
    beam_pts = np.array([800, 930, 1040, 1115, 1148, 1155, 1142, 1085, 975, 780, 510, 240, 0], dtype=float)
    keel_z_pts = np.array([-50, -120, -195, -255, -295, -320, -310, -270, -200, -110, -15, 350, 980], dtype=float)

    cs_sheer = CubicSpline(stations_x, sheer_z_pts)
    cs_beam = CubicSpline(stations_x, beam_pts)
    cs_keel = CubicSpline(stations_x, keel_z_pts)

    def beam(x):
        return max(4.0, float(cs_beam(x)))

    def deck_z(x, y):
        b = beam(x)
        return float(cs_sheer(x)) + DECK_CAMBER * (1.0 - (min(abs(y), b) / b) ** 2)

    # ------------------------------------------------------------------
    # Hull ring: keel -> starboard side -> rub rail -> flat cambered deck ->
    # rub rail -> port side -> keel. Hull part is identical to V3.
    # ------------------------------------------------------------------
    def get_sheriff_ring(x, n_hull_half=36, n_deck_half=25):
        b = beam(x)
        kz = float(cs_keel(x))
        sz = float(cs_sheer(x))
        t = x / 6000.0

        p = 1.25 + 0.55 * (t - 0.5) / 0.5 if t > 0.5 else 1.25 - 0.20 * (0.5 - t) / 0.5
        q = 1.35 - 0.25 * (t - 0.5) / 0.5 if t > 0.5 else 1.35 + 0.25 * (0.5 - t) / 0.5

        theta_s = np.linspace(0, np.pi/2 * 0.96, n_hull_half)
        ys_hull = b * (np.sin(theta_s) ** p)
        zs_hull = kz + (sz - kz) * (1.0 - np.cos(theta_s) ** q)

        rail_r = 22.0 if b > 50 else b * 0.15
        pts_rail_stbd_y = [b, b + rail_r, b]
        pts_rail_stbd_z = [sz - rail_r*0.7, sz, sz + rail_r*0.5]

        y_deck_eval = np.linspace(0.0, b, n_deck_half)
        z_deck_half = sz + DECK_CAMBER * (1.0 - (y_deck_eval / b) ** 2)

        y_deck_stbd = y_deck_eval[::-1]
        z_deck_stbd = z_deck_half[::-1]
        y_deck_port = -y_deck_eval[1:]
        z_deck_port = z_deck_half[1:]
        y_deck_full = np.concatenate([y_deck_stbd, y_deck_port])
        z_deck_full = np.concatenate([z_deck_stbd, z_deck_port])

        pts_rail_port_y = [-b, -b - rail_r, -b]
        pts_rail_port_z = [sz + rail_r*0.5, sz, sz - rail_r*0.7]

        theta_p = np.linspace(np.pi/2 * 0.96, 0, n_hull_half)
        yp_hull = -b * (np.sin(theta_p) ** p)
        zp_hull = kz + (sz - kz) * (1.0 - np.cos(theta_p) ** q)

        ring_y = np.concatenate([ys_hull, pts_rail_stbd_y, y_deck_full[1:-1], pts_rail_port_y, yp_hull[1:]])
        ring_z = np.concatenate([zs_hull, pts_rail_stbd_z, z_deck_full[1:-1], pts_rail_port_z, zp_hull[1:]])
        ring_x = np.full_like(ring_y, x)
        return np.column_stack([ring_x, ring_y, ring_z])

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

    c_idx = len(all_verts)
    all_verts = np.vstack([all_verts, np.array([[0.0, 0.0, 300.0]])])
    for im in range(m_pts):
        im_next = (im + 1) % m_pts
        all_faces.append([c_idx, im_next, im])

    b_idx = len(all_verts)
    all_verts = np.vstack([all_verts, np.array([[6000.0, 0.0, 980.0]])])
    for im in range(m_pts):
        im_next = (im + 1) % m_pts
        all_faces.append([b_idx, (nx - 1) * m_pts + im, (nx - 1) * m_pts + im_next])

    hull_deck_mesh = trimesh.Trimesh(vertices=all_verts, faces=np.array(all_faces), process=False)
    trimesh.repair.fix_winding(hull_deck_mesh)
    trimesh.repair.fix_normals(hull_deck_mesh)
    print(f"Hull & Deck Solid: Watertight={hull_deck_mesh.is_watertight}")
    mh = to_manifold(hull_deck_mesh)

    # 2. Authentic Cast-Iron Keel & Bulb (unchanged from V3)
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
    mk = to_manifold(trimesh.Trimesh(vertices=np.array(keel_verts), faces=np.array(keel_faces), process=False))

    # ------------------------------------------------------------------
    # 3. Coachroof: lofted body following the sheer line in plan view.
    #    Roof and forward slope are linear in X -> planar faces.
    # ------------------------------------------------------------------
    z_deck_bulk = deck_z(X_BULKHEAD, 0.0)
    roof_side_aft = float(cs_sheer(X_BULKHEAD)) + CABIN_SIDE_H
    roof_k = (float(cs_sheer(X_ROOF_FRONT)) - float(cs_sheer(X_BULKHEAD))) / (X_ROOF_FRONT - X_BULKHEAD)
    roof_side_front = roof_side_aft + roof_k * (X_ROOF_FRONT - X_BULKHEAD)

    def roof_side_z(x):
        if x <= X_ROOF_FRONT:
            return roof_side_aft + roof_k * (x - X_BULKHEAD)
        return roof_side_front - (x - X_ROOF_FRONT) * np.tan(SLOPE_ANGLE)

    # X where the slope reaches the deck: end of the rounded nose
    x_nose_end = X_ROOF_FRONT
    while roof_side_z(x_nose_end) > deck_z(x_nose_end, 0.0) and x_nose_end < 5500.0:
        x_nose_end += 1.0
    nose_len = x_nose_end - X_ROOF_FRONT

    def cabin_section(x, d=0.0):
        """Closed cross-section of the coachroof body, offset outward by d."""
        b = beam(x)
        y_base = b - CABIN_MARGIN + d
        # Rounded aft roof corners (plan view)
        dx_aft = min(ROOF_AFT_R, max(0.0, X_BULKHEAD + ROOF_AFT_R - x))
        corner_in = ROOF_AFT_R - np.sqrt(ROOF_AFT_R ** 2 - dx_aft ** 2)
        y_base -= corner_in
        z_base = deck_z(x, y_base)
        zrs = roof_side_z(x) + d
        h_wall = max(0.0, zrs - z_base)
        y_top = y_base - h_wall * np.tan(WALL_TUMBLEHOME)
        # Rounded nose: the roof/slope outline follows a quarter-ellipse in plan view
        if x > X_ROOF_FRONT:
            t = min(1.0, (x - X_ROOF_FRONT) / nose_len)
            y_top *= np.sqrt(max(0.0025, 1.0 - t * t))
        y_top = max(y_top, 8.0)
        z_bot = z_base - 150.0
        zrs = max(zrs, z_bot + 10.0)
        ys = np.linspace(y_top, 0.0, 6)
        zs = zrs + CABIN_ROOF_CAMBER * (1.0 - (ys / max(y_top, 1.0)) ** 2)
        stbd = [(x, y_base, z_bot)] + [(x, yy, zz) for yy, zz in zip(ys, zs)]
        port = [(x, -yy, zz) for yy, zz in zip(ys[::-1][1:], zs[::-1][1:])] + [(x, -y_base, z_bot)]
        return np.array(stbd + port)

    cab_x = np.concatenate([np.arange(X_BULKHEAD - 50.0, X_BULKHEAD + ROOF_AFT_R + 1.0, 15.0),
                            np.arange(X_BULKHEAD + ROOF_AFT_R + 50.0, X_ROOF_FRONT, 50.0), [X_ROOF_FRONT],
                            np.arange(X_ROOF_FRONT + 25.0, x_nose_end + 60.0, 25.0)])

    def cabin_body(d=0.0):
        return loft([cabin_section(x, d) for x in cab_x])

    def rake_cut(x_at_deck):
        """Half-space aft of the raked bulkhead plane passing through (x_at_deck, z_deck_bulk)."""
        cut = trimesh.creation.box([4000.0, 6000.0, 6000.0])
        cut.apply_translation([-2000.0, 0.0, 0.0])
        cut.apply_transform(trimesh.transformations.rotation_matrix(BULKHEAD_RAKE, [0, 1, 0]))
        cut.apply_translation([x_at_deck, 0.0, z_deck_bulk])
        return to_manifold(cut)

    cab = cabin_body(0.0) - rake_cut(X_BULKHEAD)

    # Companionway door: recessed outline in the aft bulkhead
    cab_door_ref = cabin_body(0.0) - rake_cut(X_BULKHEAD + DOOR_RECESS / np.cos(BULKHEAD_RAKE))
    door_box = mbox(X_BULKHEAD - 200.0, X_BULKHEAD + 400.0, -DOOR_HALF_W, DOOR_HALF_W, DOOR_Z0, DOOR_Z1)
    cab = cab - (door_box - cab_door_ref)

    # Side windows: raised trapezoid bezel + recessed tinted pane, both sides
    def win_prism(inset):
        zb = deck_z(2800.0, beam(2800.0) - CABIN_MARGIN) + WIN_Z_ABOVE_BASE + inset
        zt = zb + WIN_H - 2 * inset
        poly = np.array([
            [WIN_X_BOT[0] + inset, zb],
            [WIN_X_BOT[1] - inset, zb],
            [WIN_X_TOP[1] - inset * 0.6, zt],
            [WIN_X_TOP[0] + inset * 0.6, zt],
        ])
        sec = lambda y: np.array([[px, y, pz] for px, pz in poly])
        return loft([sec(-2500.0), sec(2500.0)])

    cab = cab + (win_prism(0.0) ^ cabin_body(WIN_FRAME_RAISE))
    cab = cab - (win_prism(WIN_FRAME_W) - cabin_body(-WIN_GLASS_RECESS))

    # Forward hatch: flat tinted acrylic plate lying on the slope
    hatch_col = mbox(HATCH_X0, HATCH_X1, -HATCH_HALF_W, HATCH_HALF_W, 0.0, 3000.0)
    cab = cab + (hatch_col ^ cabin_body(HATCH_RAISE))

    # Mast step pad at the forward edge of the flat roof
    mast_pad = trimesh.creation.cylinder(radius=MAST_PAD_R, height=60.0)
    mast_pad.apply_translation([MAST_X, 0.0, roof_side_z(MAST_X) + CABIN_ROOF_CAMBER + 10.0])
    cab = cab + to_manifold(mast_pad)

    # Toe rails along the foredeck edges, growing out of the cabin walls
    def rail_sections(sign):
        secs = []
        for x in np.arange(X_ROOF_FRONT + 100.0, RAIL_X1 + 1.0, 50.0):
            b = beam(x)
            y_out = b - CABIN_MARGIN          # outer face flush with the cabin wall foot
            y_in = y_out - RAIL_T
            t = (x - X_ROOF_FRONT) / (RAIL_X1 - X_ROOF_FRONT)
            h = RAIL_H_AFT + t * (RAIL_H_FWD - RAIL_H_AFT)
            zd = deck_z(x, y_in)
            secs.append(np.array([[x, sign * y_in, zd - 40.0], [x, sign * y_in, zd + h],
                                  [x, sign * y_out, zd + h], [x, sign * y_out, zd - 40.0]]))
        return secs
    cab = cab + loft(rail_sections(+1)) + loft(rail_sections(-1))
    cabin_solid = to_watertight_trimesh(cab)
    print(f"Coachroof Solid: Watertight={cabin_solid.is_watertight}")

    # ------------------------------------------------------------------
    # 4. Cockpit cut-outs
    # ------------------------------------------------------------------
    def rect_sec(x, hw, z0, z1):
        return np.array([[x, hw, z0], [x, hw, z1], [x, -hw, z1], [x, -hw, z0]])

    def footwell_hw(x):
        t = (x - X_COCKPIT_AFT) / (X_BULKHEAD - X_COCKPIT_AFT)
        return FOOTWELL_HW_AFT + t * (FOOTWELL_HW_FWD - FOOTWELL_HW_AFT)

    # Bench cut: follows the coaming, then turns in a quarter circle towards the bulkhead,
    # leaving a rounded shoulder at deck height where each bench meets the cabin.
    def well_hw(x):
        hw = beam(x) - GUNWALE_W
        xc = X_BULKHEAD - SEAT_END_R
        if x > xc:
            hw = min(hw, footwell_hw(X_BULKHEAD) + np.sqrt(max(0.0, SEAT_END_R ** 2 - (x - xc) ** 2)))
        return hw

    well_x = np.concatenate([np.arange(X_COCKPIT_AFT, X_BULKHEAD - SEAT_END_R, 40.0),
                             np.arange(X_BULKHEAD - SEAT_END_R, X_BULKHEAD - 1.0, 10.0), [X_BULKHEAD]])
    cockpit_well = loft([rect_sec(x, well_hw(x), SEAT_Z, 2500.0) for x in well_x])

    footwell = loft([rect_sec(X_COCKPIT_AFT, footwell_hw(X_COCKPIT_AFT), FLOOR_Z, SEAT_Z + 100.0),
                     rect_sec(X_BULKHEAD, footwell_hw(X_BULKHEAD), FLOOR_Z, SEAT_Z + 100.0)])

    # Foot-brace board on two posts + transverse vang beam
    fittings = None
    for px in POST_X:
        post = mbox(px - POST_W/2, px + POST_W/2, -POST_W/2, POST_W/2, FLOOR_Z - 20.0, BOARD_Z[1] + 10.0)
        fittings = post if fittings is None else fittings + post
    board = mbox(POST_X[0] - 60.0, POST_X[1] + 60.0, -BOARD_T/2, BOARD_T/2, BOARD_Z[0], BOARD_Z[1])
    beam_hw = footwell_hw(BEAM_X) + 40.0
    vang_beam = mbox(BEAM_X - BEAM_W/2, BEAM_X + BEAM_W/2, -beam_hw, beam_hw, BEAM_Z[0], BEAM_Z[1])
    fittings = fittings + board + vang_beam

    # Mooring cleat, athwartships, mid-foredeck
    zc = deck_z(CLEAT_X, 0.0)
    cleat = mbox(CLEAT_X - 18.0, CLEAT_X + 18.0, -30.0, 30.0, zc - 10.0, zc + 30.0) + \
            mbox(CLEAT_X - 16.0, CLEAT_X + 16.0, -70.0, 70.0, zc + 25.0, zc + 50.0)

    # ------------------------------------------------------------------
    # 5. Assemble full boat
    # ------------------------------------------------------------------
    m_full = mh + mk + cab
    m_full = m_full - cockpit_well - footwell
    m_full = m_full + fittings + cleat
    full_boat = to_watertight_trimesh(m_full)
    print(f"FULL BOAT V4 SOLID: Watertight={full_boat.is_watertight}, Volume={full_boat.volume:.1f} mm³")

    # 6. Waterline Model (DWL Z=0 cut with flat bottom)
    cut_box = trimesh.creation.box([14000.0, 6000.0, 2500.0])
    cut_box.apply_translation([3000.0, 0.0, -1250.0])
    m_wl = m_full - to_manifold(cut_box)
    waterline_boat = to_watertight_trimesh(m_wl)
    print(f"WATERLINE SOLID: Watertight={waterline_boat.is_watertight}, Volume={waterline_boat.volume:.1f} mm³")

    # 7. Split Model: Deck (flat bottom) + Keel (flat top) with 2 registration sockets
    top_box = trimesh.creation.box([14000.0, 6000.0, 3000.0])
    top_box.apply_translation([3000.0, 0.0, 1500.0])
    m_bottom = m_full - to_manifold(top_box)

    def pin_pair(depth):
        pins = None
        for px in (3800.0, 1500.0):
            c = trimesh.creation.cylinder(radius=PIN_R, height=2 * depth)
            c.apply_translation([px, 0.0, 0.0])
            pins = to_manifold(c) if pins is None else pins + to_manifold(c)
        return pins

    m_deck_split = m_wl - pin_pair(DECK_SOCKET_DEPTH)
    m_keel_split = m_bottom - pin_pair(KEEL_SOCKET_DEPTH)
    split_deck = to_watertight_trimesh(m_deck_split)
    split_keel = to_watertight_trimesh(m_keel_split)
    print(f"SPLIT DECK: Watertight={split_deck.is_watertight}")
    print(f"SPLIT KEEL: Watertight={split_keel.is_watertight}")

    # 8. Rudder & Tiller (unchanged): mounted outside the transom, tiller above the transom rim
    r_blade = trimesh.creation.box([300.0, 50.0, 950.0])
    r_blade.apply_translation([-180.0, 0.0, -100.0])
    r_cheeks = trimesh.creation.box([120.0, 75.0, 360.0])
    r_cheeks.apply_translation([-60.0, 0.0, 500.0])
    r_tiller = trimesh.creation.box([850.0, 45.0, 45.0])
    r_tiller.apply_translation([350.0, 0.0, 710.0])
    rot_p = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
    p_top = trimesh.creation.cylinder(radius=25.0, height=80.0)
    p_top.apply_transform(rot_p)
    p_top.apply_translation([10.0, 0.0, 560.0])
    p_bot = trimesh.creation.cylinder(radius=25.0, height=80.0)
    p_bot.apply_transform(rot_p)
    p_bot.apply_translation([10.0, 0.0, 360.0])
    m_rudder = to_manifold(r_blade) + to_manifold(r_cheeks) + to_manifold(r_tiller) + to_manifold(p_top) + to_manifold(p_bot)
    rudder_solid = to_watertight_trimesh(m_rudder)
    print(f"Rudder Solid: Watertight={rudder_solid.is_watertight}")

    # 9. Rigging (unchanged)
    mast_cyl = trimesh.creation.cylinder(radius=45.0, height=6800.0)
    mast_cyl.apply_translation([0.0, 0.0, 3400.0])
    boom_cyl = trimesh.creation.cylinder(radius=35.0, height=2700.0)
    boom_cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    boom_cyl.apply_translation([-1350.0, 0.0, 600.0])
    spreader = trimesh.creation.box([40.0, 950.0, 30.0])
    spreader.apply_translation([0.0, 0.0, 3200.0])
    m_rig = to_manifold(mast_cyl) + to_manifold(boom_cyl) + to_manifold(spreader)
    rigging_solid = to_watertight_trimesh(m_rig)
    print(f"Rigging Solid: Watertight={rigging_solid.is_watertight}")

    # 10. Display Cradle (unchanged)
    c_base = trimesh.creation.box([4500.0, 2200.0, 160.0])
    c_base.apply_translation([2600.0, 0.0, -930.0])
    p_fwd = trimesh.creation.box([450.0, 1900.0, 750.0])
    p_fwd.apply_translation([3800.0, 0.0, -560.0])
    p_aft = trimesh.creation.box([450.0, 1900.0, 750.0])
    p_aft.apply_translation([1400.0, 0.0, -560.0])
    m_cradle_raw = to_manifold(c_base) + to_manifold(p_fwd) + to_manifold(p_aft)
    cutter = full_boat.copy()
    cutter.apply_scale(1.03)
    m_cradle = m_cradle_raw - to_manifold(cutter)
    cradle_solid = to_watertight_trimesh(m_cradle)
    print(f"Display Cradle Solid: Watertight={cradle_solid.is_watertight}")

    # -------------------------------------------------------------
    # 11. EXPORT SUITE (Scale 1:45 for Adventurer 3 & 1:1)
    # -------------------------------------------------------------
    os.makedirs("stl_output", exist_ok=True)
    scale_1to45 = 1.0 / 45.0

    models = [
        ("Jouet_Sheriff_Waterline_1to45.stl", waterline_boat, scale_1to45, "Ватерлинейная модель (печать БЕЗ поддержек, плоское дно)"),
        ("Jouet_Sheriff_Full_1to45.stl", full_boat, scale_1to45, "Монолитный корпус с килем, рубкой и кокпитом"),
        ("Jouet_Sheriff_Split_Deck_1to45.stl", split_deck, scale_1to45, "Верхняя половина (палуба, рубка, кокпит; плоский низ со стыковочными пазами)"),
        ("Jouet_Sheriff_Split_Keel_1to45.stl", split_keel, scale_1to45, "Нижняя половина (киль и днище, плоский верх со стыковочными пазами)"),
        ("Jouet_Sheriff_Rudder_1to45.stl", rudder_solid, scale_1to45, "Навесной руль с румпелем (монтируется снаружи транца)"),
        ("Jouet_Sheriff_Rigging_1to45.stl", rigging_solid, scale_1to45, "Мачта с гиком и краспицами (печать по диагонали стола 212 мм)"),
        ("Jouet_Sheriff_Display_Cradle_1to45.stl", cradle_solid, scale_1to45, "Настольный кильблок-подставка"),
        ("Jouet_Sheriff_Full_1to1.stl", full_boat, 1.0, "Полный корпус 1:1 (LOA 6000 мм)"),
        ("Jouet_Sheriff_Waterline_1to1.stl", waterline_boat, 1.0, "Ватерлинейный корпус 1:1"),
    ]

    print("\n----------------------------------------------------------")
    print("  EXPORTING ALL STL FILES (V4)...")
    print("----------------------------------------------------------")
    all_ok = True
    for fname, m_obj, sc, desc in models:
        out = m_obj.copy()
        if sc != 1.0:
            out.apply_scale(sc)
        p = os.path.join("stl_output", fname)
        out.export(p)
        e = out.extents
        all_ok &= bool(out.is_watertight)
        print(f"[{fname}]")
        print(f"  Назначение:  {desc}")
        print(f"  Размеры:     X={e[0]:.1f} мм, Y={e[1]:.1f} мм, Z={e[2]:.1f} мм")
        print(f"  Watertight:  {out.is_watertight} (Solid manifold), Полигонов: {len(out.faces)}")
        print()

    sc_boat = full_boat.copy()
    sc_boat.apply_scale(scale_1to45)
    sc_cradle = cradle_solid.copy()
    sc_cradle.apply_scale(scale_1to45)
    sc_rudder = rudder_solid.copy()
    sc_rudder.apply_scale(scale_1to45)
    scene = trimesh.Scene([sc_boat, sc_cradle, sc_rudder])
    scene.export("stl_output/Jouet_Sheriff_Assembly_Preview.obj")
    print("Exported 3D preview scene: stl_output/Jouet_Sheriff_Assembly_Preview.obj")
    if not all_ok:
        raise SystemExit("ERROR: at least one exported model is not watertight")


if __name__ == "__main__":
    build_jouet_sheriff_v4()
