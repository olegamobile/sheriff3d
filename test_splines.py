import numpy as np
import trimesh
from scipy.interpolate import CubicSpline

# Let's verify spline interpolation for station generation
stations_x = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000], dtype=float)
# Sheer z: transom=650, mid=760, bow=980
sheer_z_pts = np.array([650, 670, 700, 725, 745, 760, 780, 810, 850, 895, 935, 960, 980], dtype=float)
sheer_spline = CubicSpline(stations_x, sheer_z_pts)

# Half-beam at sheer:
# Transom: 800, midships: 1155, bow: 0
beam_pts = np.array([820, 950, 1060, 1125, 1150, 1155, 1140, 1080, 960, 760, 500, 240, 0], dtype=float)
beam_spline = CubicSpline(stations_x, beam_pts)

# Keelson z (bottom of canoe body):
keel_z_pts = np.array([-50, -120, -200, -260, -300, -320, -310, -270, -200, -120, -30, 350, 980], dtype=float)
keel_spline = CubicSpline(stations_x, keel_z_pts)

print("Spline test successful. Max beam:", beam_spline(2500), "Sheer at bow:", sheer_spline(6000))
