import cv2
import numpy as np

im = cv2.imread("extracted_pdf_assets/Bateaux_oct69/lines_plan_oriented.png")
h, w, _ = im.shape
print(f"lines_plan_oriented shape: {w}x{h}")

# The blueprint has a blue background and white lines.
# Let's extract the white lines into a clean mask
# Blue channel is high in background, red and green are lower. White has high R, G, B.
b, g, r = cv2.split(im)
# White lines: r > 120 and g > 150
mask = (r > 120) & (g > 150)
white_lines = np.zeros((h, w), dtype=np.uint8)
white_lines[mask] = 255

cv2.imwrite("extracted_pdf_assets/Bateaux_oct69/white_lines.png", white_lines)
print("Saved white_lines.png")
