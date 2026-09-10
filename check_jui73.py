import os
import cv2

for f in sorted(os.listdir("extracted_pdf_assets/Bateaux_jui73")):
    p = os.path.join("extracted_pdf_assets/Bateaux_jui73", f)
    im = cv2.imread(p)
    # Check if rotated/flipped
    print(f"Bateaux_jui73 {f}: {im.shape}")
