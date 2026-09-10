import numpy as np
import trimesh
from scipy.interpolate import CubicSpline

def create_sheriff_mesh(scale_factor=1.0/45.0, add_keel=True, add_rudder=False):
    # Base dimensions in mm (1:1)
    stations_x = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000], dtype=float)
    sheer_z_pts = np.array([650, 675, 705, 730, 748, 760, 780, 810, 850, 895, 935, 960, 980], dtype=float)
    beam_pts = np.array([800, 930, 1040, 1115, 1148, 1155, 1142, 1085, 975, 780, 510, 240, 0], dtype=float)
    keel_z_pts = np.array([-50, -120, -195, -255, -295, -320, -310, -270, -200, -110, -15, 350, 980], dtype=float)

    cs_sheer = CubicSpline(stations_x, sheer_z_pts)
    cs_beam = CubicSpline(stations_x, beam_pts)
    cs_keel = CubicSpline(stations_x, keel_z_pts)

    # Discretization
    nx = 61  # along length
    ntheta = 25  # from keel to sheer

    x_grid = np.linspace(0, 6000, nx)
    
    # 1. Generate hull surface vertices
    # We will generate vertices for Starboard (+y) and Port (-y)
    verts_starboard = np.zeros((nx, ntheta, 3))
    
    for ix, x in enumerate(x_grid):
        b = max(0.0, float(cs_beam(x)))
        kz = float(cs_keel(x))
        sz = float(cs_sheer(x))
        t = x / 6000.0
        
        if t > 0.5:
            p = 1.25 + 0.55 * (t - 0.5) / 0.5
            q = 1.35 - 0.25 * (t - 0.5) / 0.5
        else:
            p = 1.25 - 0.20 * (0.5 - t) / 0.5
            q = 1.35 + 0.25 * (0.5 - t) / 0.5
            
        theta = np.linspace(0, np.pi/2, ntheta)
        y = b * (np.sin(theta) ** p)
        z = kz + (sz - kz) * (1.0 - np.cos(theta) ** q)
        
        verts_starboard[ix, :, 0] = x
        verts_starboard[ix, :, 1] = y
        verts_starboard[ix, :, 2] = z
        
    print(f"Generated hull grid: {nx}x{ntheta}")
    return verts_starboard

vs = create_sheriff_mesh()
print("Starboard grid shape:", vs.shape)
