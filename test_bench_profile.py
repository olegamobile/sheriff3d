import numpy as np

def get_cockpit_half(b, sz):
    sole_w = 220.0
    sole_z = 190.0
    bench_w = 400.0 # bench extends from 220 to 620 mm
    bench_z = 460.0
    coam_w = b - 180.0
    coam_z = sz + 45.0
    
    # Strictly increasing Y from 0 to b
    key_y = np.array([
        0.0,
        sole_w,
        sole_w + 15.0,
        sole_w + bench_w,
        coam_w,
        coam_w + 30.0,
        b
    ])
    key_z = np.array([
        sole_z,
        sole_z,
        bench_z,
        bench_z,
        coam_z,
        sz + 12.0,
        sz
    ])
    
    y_eval = np.linspace(0.0, b, 25)
    z_eval = np.interp(y_eval, key_y, key_z)
    return y_eval, z_eval

y, z = get_cockpit_half(1100.0, 740.0)
print("Cockpit Profile Points (Y -> Z):")
for py, pz in zip(y, z):
    print(f"  y={py:5.1f} mm : z={pz:5.1f} mm")
