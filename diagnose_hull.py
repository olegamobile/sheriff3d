import numpy as np
import trimesh
import manifold3d as m3d
from generate_sheriff_models import *

# Let's test why raw_hull_mesh gave volume=0
stations_x = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000], dtype=float)
sheer_z_pts = np.array([650, 675, 705, 730, 748, 760, 780, 810, 850, 895, 935, 960, 980], dtype=float)
beam_pts = np.array([800, 930, 1040, 1115, 1148, 1155, 1142, 1085, 975, 780, 510, 240, 0], dtype=float)
keel_z_pts = np.array([-50, -120, -195, -255, -295, -320, -310, -270, -200, -110, -15, 350, 980], dtype=float)

cs_sheer = CubicSpline(stations_x, sheer_z_pts)
cs_beam = CubicSpline(stations_x, beam_pts)
cs_keel = CubicSpline(stations_x, keel_z_pts)

nx = 61
x_vals = np.linspace(0, 6000, nx)
n_half = 28

# Let's test raw_hull_mesh before to_manifold_trimesh
print("Checking test_ring_topology vs generate_sheriff_models...")
