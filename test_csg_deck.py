import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import trimesh
import manifold3d as m3d
from scipy.interpolate import CubicSpline

# Let's test building the exact Jouët Sheriff deck and superstructure with CSG
print("Testing CSG deck architecture for Jouët Sheriff...")
