import os
import urllib.request
import re
from urllib.parse import urljoin

pdf_and_plans = [
    "http://sheriff600.chez.com/sheriff/article/Bateaux_oct69.PDF",
    "http://sheriff600.chez.com/calife/article/Bateaux_jui73.PDF",
    "http://sheriff600.chez.com/sheriff/manuel/manuel.PDF",
    "http://sheriff600.chez.com/plans/Hublot/hublot.wmf",
    "http://sheriff600.chez.com/plans/Helm/plan_safran_gunnar.wmf",
    "http://sheriff600.chez.com/plans/ferrures/ferrures_safran.wmf",
]

headers = {'User-Agent': 'Mozilla/5.0'}

os.makedirs("chez_downloads", exist_ok=True)

for url in pdf_and_plans:
    fname = os.path.basename(url)
    out_path = os.path.join("chez_downloads", fname)
    print(f"Downloading {url} ...")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp, open(out_path, 'wb') as f:
            f.write(resp.read())
        print(f"Saved {fname} ({os.path.getsize(out_path)} bytes)")
    except Exception as e:
        print(f"Failed {url}: {e}")

# Also check images in all bxp pages of sheriff/article and calife/article
for sub in ["sheriff/article", "calife/article"]:
    for i in range(1, 12):
        bxp_url = f"http://sheriff600.chez.com/{sub}/bxp{i}.html" if i > 1 else f"http://sheriff600.chez.com/{sub}/articlebx.html"
        try:
            req = urllib.request.Request(bxp_url, headers=headers)
            with urllib.request.urlopen(req) as resp:
                html = resp.read().decode('iso-8859-1', errors='ignore')
                for img in re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.I):
                    img_url = urljoin(bxp_url, img)
                    img_name = f"{sub.replace('/', '_')}_{os.path.basename(img)}"
                    img_out = os.path.join("chez_downloads", img_name)
                    if not os.path.exists(img_out):
                        req2 = urllib.request.Request(img_url, headers=headers)
                        with urllib.request.urlopen(req2) as resp2, open(img_out, 'wb') as f:
                            f.write(resp2.read())
                        print(f"Saved page img {img_name} ({os.path.getsize(img_out)} bytes)")
        except Exception as e:
            # page might not exist
            pass
