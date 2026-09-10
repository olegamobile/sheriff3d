import os
from PIL import Image

for root, dirs, files in os.walk("extracted_pdf_assets"):
    for f in files:
        path = os.path.join(root, f)
        try:
            im = Image.open(path)
            print(f"{path}: size={im.size}, mode={im.mode}, format={im.format}")
        except Exception as e:
            print(f"{path}: error {e}")
