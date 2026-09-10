import os
import cv2
import numpy as np

for doc in ["Bateaux_oct69", "Bateaux_jui73"]:
    dpath = os.path.join("extracted_pdf_assets", doc)
    files = sorted(os.listdir(dpath))
    print(f"\n--- {doc} ---")
    for f in files:
        im_path = os.path.join(dpath, f)
        img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Check mean brightness, variance, etc.
        # Let's see if there are line drawings or text
        h, w = img.shape
        print(f"{f}: shape={img.shape}")
