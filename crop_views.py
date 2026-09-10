import cv2
import numpy as np

im = cv2.imread("extracted_pdf_assets/Bateaux_oct69/lines_plan_oriented.png")
h, w, _ = im.shape

# Let's crop:
# Top part: profile view (hull profile from bow to stern, keel, rudder)
# Bottom part: half-breadth plan view (waterlines)
# Middle right: body plan (transverse station curves)

# Let's save crops to inspect their bounds
profile = im[10:195, 20:530]
half_breadth = im[185:380, 50:520]
body_plan = im[130:280, 460:650]

cv2.imwrite("extracted_pdf_assets/Bateaux_oct69/crop_profile.png", profile)
cv2.imwrite("extracted_pdf_assets/Bateaux_oct69/crop_half_breadth.png", half_breadth)
cv2.imwrite("extracted_pdf_assets/Bateaux_oct69/crop_body_plan.png", body_plan)

print("Crops saved successfully.")
