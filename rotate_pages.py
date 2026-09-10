import os
import cv2

for doc in ["Bateaux_oct69"]:
    dpath = os.path.join("extracted_pdf_assets", doc)
    for f in os.listdir(dpath):
        p = os.path.join(dpath, f)
        im = cv2.imread(p)
        if im is not None:
            im = cv2.rotate(im, cv2.ROTATE_180)
            cv2.imwrite(p, im)
print("Rotated Bateaux_oct69 180 degrees.")
