# Coordinate and scale calibration
# Real boat dimensions (Jouët Sheriff 600):
# LOA: 6.00 m (6000 mm)
# LWL: 5.00 m (5000 mm)
# Beam max: 2.31 m (2310 mm)
# Draft: 0.85 m (850 mm)
# Freeboard bow: 0.98 m (980 mm)
# Freeboard midships: 0.76 m (760 mm)
# Freeboard stern: 0.65 m (650 mm)
# Keel depth below canoe body: approx 0.55 m (550 mm)
# Canoe body depth below WL: approx 0.30 m (300 mm)

# Flashforge Adventurer 3 scale: 1:45
scale = 1.0 / 45.0
print(f"Scale 1:45 target dimensions:")
print(f"  Length: {6000 * scale:.1f} mm")
print(f"  Beam:   {2310 * scale:.1f} mm")
print(f"  Draft:  {850 * scale:.1f} mm")
print(f"  Total Height (keel to coachroof): ~{(850 + 760 + 450) * scale:.1f} mm")
