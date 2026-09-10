import numpy as np
import trimesh

print("Testing trimesh capabilities...")
# Test basic solid creation and boolean / slicing
box = trimesh.creation.box(extents=[10, 10, 10])
print(f"Box watertight: {box.is_watertight}, volume: {box.volume}")
