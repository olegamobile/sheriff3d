import cv2
import numpy as np

bp = cv2.imread("extracted_pdf_assets/Bateaux_oct69/crop_body_plan.png")
h, w, _ = bp.shape

# Centerline is x=90, WL is y=73
# Body plan shows:
# Left half (x < 90): aft stations 0, 1, 2, 3, 4, 5
# Right half (x > 90): forward stations 6, 7, 8, 9, 10
# Let's inspect brightness profile along horizontal lines at various heights (waterlines):
# y=33 (sheer), y=53 (WL+1), y=73 (DWL), y=93 (WL-1 / canoe body keel)

gray = cv2.cvtColor(bp, cv2.COLOR_BGR2GRAY)

print("Body plan shape:", w, h)
for y in [30, 45, 60, 73, 85, 95]:
    row = gray[y, :]
    peaks_left = np.where((row > 140) & (np.arange(w) < 88))[0]
    peaks_right = np.where((row > 140) & (np.arange(w) > 92))[0]
    print(f"y={y}: left peaks at {peaks_left}, right peaks at {peaks_right}")
