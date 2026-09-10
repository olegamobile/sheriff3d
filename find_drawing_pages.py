import cv2
import numpy as np

for i in range(1, 11):
    path = f"extracted_pdf_assets/Bateaux_jui73/p{i}_corrected.png"
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Check if there are large drawings (drawings have long horizontal and vertical lines)
    edges = cv2.Canny(im, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    line_count = len(lines) if lines is not None else 0
    print(f"Page {i}: long lines count = {line_count}")
