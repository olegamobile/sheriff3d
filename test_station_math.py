import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Let's calibrate station curves against the blueprint body plan
# Blueprint body plan has width 190, height 150
# Centerline is x=90, WL is y=73, baseline is y=92, sheer is y=33
# Scale: 1 pixel in body plan = ~12.3 mm

stations_x = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000], dtype=float)
sheer_z_pts = np.array([650, 675, 705, 730, 748, 760, 780, 810, 850, 895, 935, 960, 980], dtype=float)
beam_pts = np.array([800, 930, 1040, 1115, 1148, 1155, 1142, 1085, 975, 780, 510, 240, 0], dtype=float)
keel_z_pts = np.array([-50, -120, -195, -255, -295, -320, -310, -270, -200, -110, -15, 350, 980], dtype=float)

cs_sheer = CubicSpline(stations_x, sheer_z_pts)
cs_beam = CubicSpline(stations_x, beam_pts)
cs_keel = CubicSpline(stations_x, keel_z_pts)

def get_station_curve(x, n_pts=25):
    b = cs_beam(x)
    kz = cs_keel(x)
    sz = cs_sheer(x)
    if b <= 0 or sz <= kz:
        return np.zeros((n_pts, 3))
    
    # Exponent variation along x
    # Forward: sharp V (p>1.4, q~1.1)
    # Aft: flat bottom, firm bilge (p~1.1, q~1.6)
    t = x / 6000.0
    if t > 0.5: # forward
        p = 1.25 + 0.6 * (t - 0.5) / 0.5
        q = 1.35 - 0.3 * (t - 0.5) / 0.5
    else: # aft
        p = 1.25 - 0.2 * (0.5 - t) / 0.5
        q = 1.35 + 0.3 * (0.5 - t) / 0.5
        
    theta = np.linspace(0, np.pi/2, n_pts)
    y = b * (np.sin(theta) ** p)
    z = kz + (sz - kz) * (1.0 - np.cos(theta) ** q)
    pts = np.zeros((n_pts, 3))
    pts[:, 0] = x
    pts[:, 1] = y
    pts[:, 2] = z
    return pts

# Test midship station 5 (x=2500)
st5 = get_station_curve(2500)
print(f"Station 5: Keel at (y={st5[0,1]:.1f}, z={st5[0,2]:.1f}), Sheer at (y={st5[-1,1]:.1f}, z={st5[-1,2]:.1f})")
print("Midship Beam:", st5[-1,1] * 2, "Depth:", st5[-1,2] - st5[0,2])
