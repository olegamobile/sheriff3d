import urllib.request
import re
from urllib.parse import urljoin

base_url = "http://sheriff600.chez.com/plans/plans.html"
headers = {'User-Agent': 'Mozilla/5.0'}

req = urllib.request.Request(base_url, headers=headers)
with urllib.request.urlopen(req) as resp:
    html = resp.read().decode('iso-8859-1', errors='ignore')

print("--- PLANS PAGE LINKS & IMGS ---")
for m in re.findall(r'(?:href|src)=["\']([^"\']+)["\']', html, re.I):
    full_url = urljoin(base_url, m)
    print(full_url)
