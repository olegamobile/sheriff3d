import urllib.request
import re
from urllib.parse import urljoin

urls = [
    "http://sheriff600.chez.com/sheriff/article/articlebx.html",
    "http://sheriff600.chez.com/calife/article/articlebx.html",
    "http://sheriff600.chez.com/sheriff/manuel/manuel.html",
    "http://sheriff600.chez.com/sheriff/sheriff.html",
    "http://sheriff600.chez.com/calife/calife.html",
    "http://sheriff600.chez.com/index.html",
    "http://sheriff600.chez.com/photos/photos.html"
]

headers = {'User-Agent': 'Mozilla/5.0'}
all_assets = set()

for u in urls:
    try:
        req = urllib.request.Request(u, headers=headers)
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode('iso-8859-1', errors='ignore')
            for m in re.findall(r'(?:href|src)=["\']([^"\']+)["\']', html, re.I):
                full = urljoin(u, m)
                if any(ext in full.lower() for ext in ['.jpg', '.gif', '.png', '.wmf', '.pdf', '.bmp', '.tif', '.html']):
                    if 'sheriff600.chez.com' in full:
                        all_assets.add(full)
    except Exception as e:
        print(f"Error {u}: {e}")

print("--- FOUND ON CHEZ.COM ---")
for a in sorted(all_assets):
    print(a)
