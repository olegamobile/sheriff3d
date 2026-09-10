import cv2
import glob

print("Scanning pages for drawings/schematics...")
for doc in ["Bateaux_oct69", "Bateaux_jui73"]:
    files = sorted(glob.glob(f"extracted_pdf_assets/{doc}/*.png"))
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        # Check edges using Canny
        edges = cv2.Canny(im, 50, 150)
        edge_ratio = edges.sum() / (edges.shape[0] * edges.shape[1])
        print(f"{f}: shape={im.shape}, edge_density={edge_ratio:.2f}")
