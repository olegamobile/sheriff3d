import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import trimesh
from scipy.interpolate import CubicSpline

stations_x = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000], dtype=float)
sheer_z_pts = np.array([650, 675, 705, 730, 748, 760, 780, 810, 850, 895, 935, 960, 980], dtype=float)
beam_pts = np.array([800, 930, 1040, 1115, 1148, 1155, 1142, 1085, 975, 780, 510, 240, 0], dtype=float)
keel_z_pts = np.array([-50, -120, -195, -255, -295, -320, -310, -270, -200, -110, -15, 350, 980], dtype=float)

cs_sheer = CubicSpline(stations_x, sheer_z_pts)
cs_beam = CubicSpline(stations_x, beam_pts)
cs_keel = CubicSpline(stations_x, keel_z_pts)

def get_deck_profile(x, b, sz):
    """
    Returns perfectly symmetrical (y_coords, z_coords) across the deck from +b (starboard) to -b (port).
    """
    if x >= 4100:
        # Foredeck: smooth cambered surface
        y_pts = np.linspace(b, -b, 31)
        z_pts = []
        for y in y_pts:
            camber = 35.0 * (1.0 - (y / max(1.0, b))**2)
            z = sz + camber
            # Forward hatch (Panneau ouvrant) between x=4200..4700, |y| <= 220
            if 4200 <= x <= 4700:
                hatch_w = 220.0 - 40.0 * (x - 4200) / 500.0 # slight taper forward
                if abs(y) <= hatch_w:
                    # Raised hatch frame (+20 mm) with slight bevel
                    edge_dist = hatch_w - abs(y)
                    hatch_lift = min(20.0, edge_dist * 0.8)
                    z += hatch_lift
            z_pts.append(z)
        return y_pts, np.array(z_pts)

    elif x >= 2000:
        # Coachroof (Рубка)
        # Roof width and height
        # Forward slope from x=3400 down to x=4100
        if x >= 3400:
            slope_t = (x - 3400) / 700.0 # 0 at 3400, 1 at 4100
            roof_h = (1.0 - slope_t) * 380.0 + slope_t * 30.0
            cab_w = (1.0 - slope_t * 0.25) * (0.68 * b)
        else:
            roof_h = 380.0
            cab_w = 0.68 * b

        side_deck_w = b - cab_w
        cab_roof_z = sz + roof_h

        # Key points on starboard (from +b to 0)
        # 1. Sheer (+b, sz)
        # 2. Side deck (+cab_w + 30, sz + 12)
        # 3. Base of cabin side (+cab_w, sz + 15)
        # 4. Top of cabin side (+cab_w * 0.90, cab_roof_z - 15)
        # 5. Roof edge (+cab_w * 0.85, cab_roof_z)
        # 6. Roof crown (0, cab_roof_z + 25)
        key_y = np.array([b, cab_w + 30.0, cab_w, cab_w * 0.90, cab_w * 0.85, 0.0])
        key_z = np.array([sz, sz + 12.0, sz + 15.0, cab_roof_z - 15.0, cab_roof_z, cab_roof_z + 25.0])

        # Forward hatch if it extends into slope (x between 3700 and 4100)
        if 3650 <= x <= 4100:
            hatch_w = 230.0
            # Check hatch frame on forward slope
            key_z[5] += 15.0
            key_z[4] += 5.0

        # Symmetrical interpolation across 31 points
        # Sample starboard
        y_half = np.linspace(b, 0.0, 16)
        z_half = np.interp(y_half, key_y[::-1], key_z[::-1])

        # Mirror for port
        y_full = np.concatenate([y_half, -y_half[1:]])
        z_full = np.concatenate([z_half, z_half[1:]])
        return y_full, z_full

    else:
        # Cockpit region: x from 0 to 2000
        # Distinct benches on port and starboard, lower footwell in center
        sole_w = 240.0 # footwell half-width = 240 mm (total 480 mm)
        sole_z = 180.0 # footwell floor height
        bench_z = 460.0 # bench seat height (280 mm above footwell!)
        coam_h = sz + 45.0 # coaming top height

        # Coaming position
        coam_w = b - 200.0

        if x < 300:
            # Aft deck behind cockpit
            y_pts = np.linspace(b, -b, 31)
            camber = 15.0 * (1.0 - (y_pts / max(1.0, b))**2)
            return y_pts, sz + camber
        elif x < 600:
            # Aft locker bench (across the stern)
            # Center is at bench_z (460 mm), sides at bench_z, coaming outside
            key_y = np.array([b, coam_w + 30.0, coam_w, sole_w + 150.0, sole_w, 0.0])
            key_z = np.array([sz, sz + 15.0, coam_h, bench_z + 10.0, bench_z, bench_z])
            y_half = np.linspace(b, 0.0, 16)
            z_half = np.interp(y_half, key_y[::-1], key_z[::-1])
            y_full = np.concatenate([y_half, -y_half[1:]])
            z_full = np.concatenate([z_half, z_half[1:]])
            return y_full, z_full
        else:
            # Main cockpit with port & starboard seating benches and sunken floor
            key_y = np.array([
                b,              # sheer
                coam_w + 30.0,  # side deck outer
                coam_w,         # coaming top
                sole_w + 220.0, # bench outer edge
                sole_w + 10.0,  # bench inner edge
                sole_w,         # footwell side wall
                0.0             # footwell center
            ])
            key_z = np.array([
                sz,             # sheer
                sz + 12.0,      # side deck
                coam_h,         # coaming top (+45 mm above deck)
                bench_z,        # bench seat (+460 mm)
                bench_z,        # bench seat (+460 mm)
                sole_z,         # footwell sole (+180 mm)
                sole_z          # footwell sole (+180 mm)
            ])
            y_half = np.linspace(b, 0.0, 16)
            z_half = np.interp(y_half, key_y[::-1], key_z[::-1])
            y_full = np.concatenate([y_half, -y_half[1:]])
            z_full = np.concatenate([z_half, z_half[1:]])
            return y_full, z_full

print("Testing deck profile generator...")
y_c, z_c = get_deck_profile(1200, 1080, 720)
print("Cockpit section at x=1200:")
print("  Center sole z:", z_c[15], "(expected 180.0)")
print("  Bench seat z:", z_c[12], "(expected 460.0)")
print("  Coaming top z:", z_c[3], "Sheer z:", z_c[0])
