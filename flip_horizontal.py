import os
import cv2

for f in os.listdir("extracted_pdf_assets/Bateaux_oct69"):
    p = os.path.join("extracted_pdf_assets/Bateaux_oct69", f)
    im = cv2.imread(p)
    if im is not None:
        im = cv2.flip(im, 1)
        cv2.imwrite(p, im)
print("Flipped Bateaux_oct69 horizontally.")
