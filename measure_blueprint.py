import cv2
import numpy as np

# Let's inspect the dimensions and stations in the blueprint
im = cv2.imread("extracted_pdf_assets/Bateaux_oct69/lines_plan_oriented.png")

# Let's locate the baseline, waterline, stem, stern in the profile view
# Waterline is a horizontal line across the drawing.
# Station lines are vertical lines spaced across the hull.
# Let's analyze vertical and horizontal projections in crop_profile.png
prof = cv2.imread("extracted_pdf_assets/Bateaux_oct69/crop_profile.png")
gray = cv2.cvtColor(prof, cv2.COLOR_BGR2GRAY)
# white lines mask
mask = gray > 140

# Horizontal line detection (waterlines):
row_sums = mask.sum(axis=1)
# Vertical line detection (stations):
col_sums = mask.sum(axis=0)

print("Profile shape:", prof.shape)
# Top station peaks
peaks_x = np.where(col_sums > 30)[0]
print("Active X range:", peaks_x.min(), "to", peaks_x.max())
