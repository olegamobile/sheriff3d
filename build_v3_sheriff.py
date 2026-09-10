import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import trimesh
import manifold3d as m3d
from scipy.interpolate import CubicSpline

print("Constructing Jouët Sheriff V3 with exact planar geometry and authentic features...")
