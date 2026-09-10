import trimesh

b1 = trimesh.creation.box([10, 10, 10])
b2 = trimesh.creation.box([5, 5, 10])
b2.apply_translation([0, 0, 5])

try:
    u = b1.union(b2)
    print(f"Trimesh union success: watertight={u.is_watertight}, volume={u.volume}")
except Exception as e:
    print(f"Trimesh union error: {e}")
