import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import trimesh
import manifold3d as m3d
from scipy.interpolate import CubicSpline

# Let's create the full-beam deck and superstructure with exact planar geometry
stations_x = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000], dtype=float)
sheer_z_pts = np.array([650, 675, 705, 730, 748, 760, 780, 810, 850, 895, 935, 960, 980], dtype=float)
beam_pts = np.array([800, 930, 1040, 1115, 1148, 1155, 1142, 1085, 975, 780, 510, 240, 0], dtype=float)

cs_sheer = CubicSpline(stations_x, sheer_z_pts)
cs_beam = CubicSpline(stations_x, beam_pts)

def create_deck_superstructure():
    # Build upper deck/cabin block from sheer up to roof
    # X stations from 0 to 6000
    nx = 61
    x_grid = np.linspace(0, 5990, nx)
    
    verts = []
    faces = []
    
    # We will loft the top surface of the boat:
    # At each X:
    # from starboard sheer (+b, sz) to port sheer (-b, sz)
    # With exact planar regions:
    # 1. Foredeck (x >= 3700): smooth cambered deck
    # 2. Forward slope of coachroof (3100 <= x < 3700): planar slope!
    # 3. Main coachroof (2000 <= x < 3100): flat roof with subtle crown!
    # 4. Cockpit area (250 <= x < 2000): side decks / coaming top
    # 5. Aft deck (x < 250): stern deck
    
    n_pts_across = 31
    mesh_rings = []
    
    for ix, x in enumerate(x_grid):
        b = max(4.0, float(cs_beam(x)))
        sz = float(cs_sheer(x))
        y_vals = np.linspace(b, -b, n_pts_across)
        z_vals = []
        
        if x >= 3700:
            # Foredeck: pure camber
            camber = 35.0 * (1.0 - (y_vals / max(1.0, b))**2)
            z_vals = sz + camber
        elif x >= 3100:
            # Forward slope of coachroof: pure planar slope!
            # Slope parameter t_s: 0 at x=3100 (roof top z=sz+380), 1 at x=3700 (foredeck z=sz+30)
            t_s = (x - 3100.0) / 600.0
            roof_z = (1.0 - t_s) * (sz + 380.0) + t_s * (sz + 30.0)
            camber = (1.0 - t_s) * 20.0 * (1.0 - (y_vals / max(1.0, b))**2)
            z_vals = roof_z + camber
        elif x >= 2000:
            # Main coachroof: full beam, top is at sz + 380
            roof_z = sz + 380.0
            camber = 22.0 * (1.0 - (y_vals / max(1.0, b))**2)
            z_vals = roof_z + camber
        else:
            # Cockpit sheer/deck level: sz
            camber = 12.0 * (1.0 - (y_vals / max(1.0, b))**2)
            z_vals = sz + camber
            
        ring = np.column_stack([np.full_like(y_vals, x), y_vals, z_vals])
        mesh_rings.append(ring)
        
    print(f"Lofted deck surface: {len(mesh_rings)} stations")
    return mesh_rings

create_deck_superstructure()
