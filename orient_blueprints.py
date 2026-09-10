import cv2

im = cv2.imread("extracted_pdf_assets/Bateaux_oct69/p9_1__im9.png")
im_rot = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("extracted_pdf_assets/Bateaux_oct69/lines_plan_oriented.png", im_rot)

im10 = cv2.imread("extracted_pdf_assets/Bateaux_oct69/p10_1__im10.png")
im10_rot = cv2.rotate(im10, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("extracted_pdf_assets/Bateaux_oct69/arrangement_plan_oriented.png", im10_rot)
print("Rotated blueprints 90 clockwise.")
