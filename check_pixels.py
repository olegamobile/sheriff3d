import cv2
import numpy as np

prof = cv2.imread("extracted_pdf_assets/Bateaux_oct69/crop_profile.png")
print("BGR min/max/mean:", prof.min(axis=(0,1)), prof.max(axis=(0,1)), prof.mean(axis=(0,1)))

# Blueprint line is white/light blue, background is medium blue
# Let's inspect brightness (V in HSV or gray)
gray = cv2.cvtColor(prof, cv2.COLOR_BGR2GRAY)
print("Gray min, max, median, 95th percentile:", np.min(gray), np.max(gray), np.median(gray), np.percentile(gray, 95))
