import cv2
import numpy as np

def annotate_grid(img_path, out_path, step=25):
    im = cv2.imread(img_path)
    h, w, _ = im.shape
    annotated = im.copy()
    
    # Draw faint grid
    for x in range(0, w, step):
        cv2.line(annotated, (x, 0), (x, h), (0, 255, 255), 1)
        cv2.putText(annotated, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        
    for y in range(0, h, step):
        cv2.line(annotated, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(annotated, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        
    cv2.imwrite(out_path, annotated)

annotate_grid("extracted_pdf_assets/Bateaux_oct69/crop_profile.png", "extracted_pdf_assets/Bateaux_oct69/annotated_profile.png", 25)
annotate_grid("extracted_pdf_assets/Bateaux_oct69/crop_half_breadth.png", "extracted_pdf_assets/Bateaux_oct69/annotated_half_breadth.png", 25)
annotate_grid("extracted_pdf_assets/Bateaux_oct69/crop_body_plan.png", "extracted_pdf_assets/Bateaux_oct69/annotated_body_plan.png", 20)
print("Annotated grids saved.")
