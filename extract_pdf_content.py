import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pypdf import PdfReader


pdfs = ["Bateaux_oct69.PDF", "Bateaux_jui73.PDF", "manuel.PDF"]

for pdf_name in pdfs:
    pdf_path = os.path.join("chez_downloads", pdf_name)
    if not os.path.exists(pdf_path):
        continue
    reader = PdfReader(pdf_path)
    print(f"\n======================================")
    print(f"PDF: {pdf_name} (pages: {len(reader.pages)})")
    
    out_dir = os.path.join("extracted_pdf_assets", os.path.splitext(pdf_name)[0])
    os.makedirs(out_dir, exist_ok=True)
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            print(f"  Page {i+1} text snippet: {text.strip()[:100].replace('\n', ' ')}")
        img_count = 0
        for img_name, img_file_obj in page.images.items():
            img_count += 1
            safe_name = os.path.basename(img_name.replace("/", "_"))
            if not any(safe_name.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                safe_name += ".png"
            target_file = os.path.join(out_dir, f"p{i+1}_{img_count}_{safe_name}")
            with open(target_file, "wb") as f:
                f.write(img_file_obj.data)

        if img_count > 0:
            print(f"  Page {i+1}: extracted {img_count} images")
