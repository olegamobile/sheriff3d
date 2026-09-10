import os
import cv2

# Check all pages of Bateaux_jui73
for i in range(1, 11):
    path = f"extracted_pdf_assets/Bateaux_jui73/p{i}_1__im{i}.png"
    if os.path.exists(path):
        im = cv2.imread(path)
        # Flip horizontally if mirrored
        # Notice text in p9 was reversed horizontally
        im_flipped = cv2.flip(im, 1)
        cv2.imwrite(f"extracted_pdf_assets/Bateaux_jui73/p{i}_corrected.png", im_flipped)
print("Corrected Bateaux_jui73 pages.")
