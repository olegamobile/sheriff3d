import numpy as np
import trimesh

# Let's inspect edges of hull_solid
# A mesh is watertight if every edge has exactly 2 faces.
from generate_sheriff_models import *

# Let's run the generator step by step
stations_x = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000], dtype=float)
sheer_z_pts = np.array([650, 675, 705, 730, 748, 760, 780, 810, 850, 895, 935, 960, 980], dtype=float)
beam_pts = np.array([800, 930, 1040, 1115, 1148, 1155, 1142, 1085, 975, 780, 510, 240, 0], dtype=float)
keel_z_pts = np.array([-50, -120, -195, -255, -295, -320, -310, -270, -200, -110, -15, 350, 980], dtype=float)

cs_sheer = CubicSpline(stations_x, sheer_z_pts)
cs_beam = CubicSpline(stations_x, beam_pts)
cs_keel = CubicSpline(stations_x, keel_z_pts)

nx = 61
ntheta = 25
x_grid = np.linspace(0, 6000, nx)
n_total_theta = 2 * ntheta - 1

# Let's check boundary edges of hull_skin alone
# In hull_skin, open edges should ONLY be:
# - port sheer (it = 0)
# - starboard sheer (it = n_total_theta - 1)
# - transom top edge
# Everything else should be closed!
